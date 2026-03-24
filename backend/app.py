import os
import io
import math
import base64
import sqlite3
import secrets
import logging
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import bcrypt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as _TVT          
import torchvision.models as _TVM             
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp

from flask import Flask, send_from_directory, jsonify, request, g
from flask_cors import CORS

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
IMAGE_SIZE = 512

CLASS_NAMES = [
    "urban_land",   # 0
    "agriculture",  # 1
    "rangeland",    # 2
    "forest",       # 3
    "water",        # 4
    "barren",       # 5
    "unknown",      # 6
]
NUM_CLASSES = len(CLASS_NAMES)
SKIP_CLASSES = {6}

CLASS_COLORS_BGR = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (0, 100, 0),
    4: (255, 0, 0),
    5: (128, 128, 255),
    6: (80, 80, 80),
}

MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.55"))
MIN_AREA       = int(os.environ.get("MIN_AREA", "2000"))
NMS_IOU_THRESH = float(os.environ.get("NMS_IOU_THRESH", "0.5"))
MAX_DETECTIONS = int(os.environ.get("MAX_DETECTIONS", "25"))


# ==============================================================================
# 1. DATABASE & AUTHENTICATION MODULE 
# ==============================================================================
def get_db():
    if "db" not in g:
        from flask import current_app
        db_path = current_app.config["DATABASE"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

def init_db(app):
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        db.commit()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False

def create_session(user_id: int) -> str:
    token      = secrets.token_urlsafe(32)
    created_at = datetime.utcnow().isoformat() + "Z"
    db = get_db()
    db.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, created_at),
    )
    db.commit()
    return token

def get_user_by_token(token: str):
    if not token:
        return None
    db  = get_db()
    row = db.execute(
        "SELECT u.id, u.name, u.email FROM users u "
        "JOIN sessions s ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    ).fetchone()
    return dict(row) if row else None

def token_required(fn):
    def wrapper(*args, **kwargs):
        auth  = request.headers.get("Authorization", "")
        token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else None
        user  = get_user_by_token(token)
        if user is None:
            return jsonify({"error": "Unauthorized"}), 401
        request.current_user = user
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ==============================================================================
# 2. SATELLITE TILE FETCHING MODULE  
# ==============================================================================
def _latlon_to_web_mercator(lat: float, lon: float):
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y

def fetch_arcgis_tile(lat: float, lon: float, radius_m: float,
                      size: int = 512, api_key: str = None) -> bytes:
    ARCGIS_EXPORT_URL = (
        "https://services.arcgisonline.com/ArcGIS/rest/services"
        "/World_Imagery/MapServer/export"
    )
    center_x, center_y = _latlon_to_web_mercator(lat, lon)
    bbox   = [center_x-radius_m, center_y-radius_m,
               center_x+radius_m, center_y+radius_m]
    params = {
        "bbox":    ",".join(str(v) for v in bbox),
        "bboxSR":  3857,
        "imageSR": 3857,
        "size":    f"{size},{size}",
        "format":  "png",
        "f":       "image",
        "dpi":     96,
    }
    if api_key:
        params["token"] = api_key
    resp = requests.get(ARCGIS_EXPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.content


# ==============================================================================
# 3. BOUNDING BOX ENGINE MODULE 
# ==============================================================================
_PREPROCESS = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class BBoxEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str          = "cuda",
        min_confidence: float = 0.45,
        min_area_px: int      = 400,
        nms_iou_thresh: float = 0.45,
        mask_alpha: float     = 0.40,
        box_thickness: int    = 2,
        font_scale: float     = 0.55,
        max_detections: Optional[int] = None,
    ):
        self.model          = model
        self.device         = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_conf       = min_confidence
        self.min_area       = min_area_px
        self.nms_thresh     = nms_iou_thresh
        self.mask_alpha     = mask_alpha
        self.box_thickness  = box_thickness
        self.font_scale     = font_scale
        self.max_detections = max_detections
        self.model.eval()

    def predict_bgr(self, bgr: np.ndarray) -> dict:
        orig_h, orig_w     = bgr.shape[:2]
        rgb                = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        class_mask, probs  = self._run_model(rgb, orig_h, orig_w)
        detections         = self._mask_to_bboxes(class_mask, probs)
        detections         = self._nms(detections)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        if self.max_detections and len(detections) > self.max_detections:
            detections = sorted(
                detections,
                key=lambda d: (d.get("area", 0), d.get("confidence", 0)),
                reverse=True,
            )[:self.max_detections]
        mask_bgr      = self._colorize_mask(class_mask, orig_h, orig_w)
        annotated_bgr = self._draw_all(bgr, mask_bgr, detections)
        annotated_b64 = self._encode_b64(annotated_bgr)
        mask_b64      = self._encode_b64(mask_bgr)
        summary = {name: 0 for name in CLASS_NAMES}
        for d in detections:
            summary[d["class_name"]] += 1
        return {
            "detections":    detections,
            "class_mask":    class_mask,
            "annotated_b64": annotated_b64,
            "mask_b64":      mask_b64,
            "summary":       summary,
        }

    @torch.inference_mode()
    def _run_model(self, rgb, orig_h, orig_w):
        resized   = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor    = _PREPROCESS(image=resized)["image"].unsqueeze(0).to(self.device)
        use_amp   = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = self.model(tensor)
        logits_up  = F.interpolate(logits.float(), size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False)
        probs      = F.softmax(logits_up, dim=1).squeeze(0).cpu().numpy()
        class_mask = probs.argmax(axis=0).astype(np.int32)
        return class_mask, probs

    def _mask_to_bboxes(self, class_mask, probs):
        detections = []
        for cid in range(NUM_CLASSES):
            if cid in SKIP_CLASSES:
                continue
            binary = (class_mask == cid).astype(np.uint8)
            if binary.sum() < self.min_area:
                continue
            kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_clean, connectivity=8
            )
            for lid in range(1, n_labels):
                area = int(stats[lid, cv2.CC_STAT_AREA])
                if area < self.min_area:
                    continue
                x = int(stats[lid, cv2.CC_STAT_LEFT])
                y = int(stats[lid, cv2.CC_STAT_TOP])
                w = int(stats[lid, cv2.CC_STAT_WIDTH])
                h = int(stats[lid, cv2.CC_STAT_HEIGHT])
                comp_pixels = (labels == lid)
                mean_conf   = float(probs[cid][comp_pixels].mean())
                if mean_conf < self.min_conf:
                    continue
                comp_mask = comp_pixels.astype(np.uint8)
                cnts, _   = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    rx, ry, rw, rh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                    x, y, w, h = rx, ry, rw, rh
                detections.append({
                    "class_id":   cid,
                    "class_name": CLASS_NAMES[cid],
                    "confidence": round(mean_conf, 4),
                    "bbox":       [x, y, x + w, y + h],
                    "area":       area,
                })
        return detections

    def _nms(self, detections):
        if not detections:
            return []
        out = []
        for cid in set(d["class_id"] for d in detections):
            dets = sorted(
                [d for d in detections if d["class_id"] == cid],
                key=lambda d: d["confidence"], reverse=True
            )
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                dets = [d for d in dets if self._iou(best["bbox"], d["bbox"]) < self.nms_thresh]
            out.extend(keep)
        return out

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter    = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6)

    def _colorize_mask(self, class_mask, orig_h, orig_w):
        color_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cid, color in CLASS_COLORS_BGR.items():
            color_img[class_mask == cid] = color
        return color_img

    def _draw_all(self, bgr, mask_bgr, detections):
        canvas = cv2.addWeighted(bgr, 1-self.mask_alpha, mask_bgr, self.mask_alpha, 0)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cid    = det["class_id"]
            color  = CLASS_COLORS_BGR[cid]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)
            label  = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            pad    = 4
            chip_y1 = max(y1 - th - 2*pad, 0)
            chip_y2 = chip_y1 + th + 2*pad
            cv2.rectangle(canvas, (x1, chip_y1), (x1+tw+2*pad, chip_y2), color, -1)
            brightness = int(color[0])*.114 + int(color[1])*.587 + int(color[2])*.299
            text_color = (0,0,0) if brightness > 130 else (255,255,255)
            cv2.putText(canvas, label, (x1+pad, chip_y2-pad-bl//2),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, text_color, 1, cv2.LINE_AA)
        return canvas

    def _encode_b64(self, bgr, quality=90):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


# ==============================================================================
# 4. MODEL LOADER — SEGMENTATION + SATELLITE GATE
# ==============================================================================
_model_instance  = None
_sat_classifier  = None                          
_device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Gate transform  ───────────────────────────────
_GATE_TRANSFORM = _TVT.Compose([                
    _TVT.Resize((224, 224)),
    _TVT.ToTensor(),
    _TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_sat_classifier():                        
    """
    Lazy-loads satellite_classifier_v4.pth (EfficientNet-B0 binary gate).
    Returns (model, threshold) tuple.
    Cached as singleton — loaded only once per process.
    """
    global _sat_classifier
    if _sat_classifier is not None:
        return _sat_classifier

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path  = os.path.join(repo_root, "satellite_classifier_v4.pth")

    gate = _TVM.efficientnet_b0(weights=None)
    gate.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, 1),
    )

    threshold = 0.5
    if os.path.exists(ckpt_path):
        ckpt      = torch.load(ckpt_path, map_location=_device, weights_only=False)
        gate.load_state_dict(ckpt["model_state_dict"])
        threshold = ckpt.get("threshold", 0.5)
        logging.info(
            f"Satellite gate v4 loaded — "
            f"val_acc={ckpt.get('val_acc','?')}  "
            f"threshold={threshold}  "
            f"sat_train={ckpt.get('balance',{}).get('sat_train','?')}  "
            f"nonsat_train={ckpt.get('balance',{}).get('nonsat_train','?')}"
        )
    else:
        logging.warning(
            f"satellite_classifier_v4.pth not found at {ckpt_path}. "
            f"Gate disabled — all images will pass to segmenter."
        )

    gate = gate.to(_device)
    gate.eval()
    _sat_classifier = (gate, threshold)
    return _sat_classifier


@torch.inference_mode()
def is_satellite_image(pil_img: Image.Image) -> tuple: 
    """
    Runs the binary gate classifier on a PIL image.
    Returns (is_satellite: bool, score: float).
      score close to 1.0 = confident satellite
      score close to 0.0 = confident non-satellite
    """
    gate, threshold = get_sat_classifier()
    tensor  = _GATE_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(_device)
    use_amp = _device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp):
        logit = gate(tensor)
    score = float(torch.sigmoid(logit).squeeze())
    return score >= threshold, round(score, 4)


def get_model():
    global _model_instance
    if _model_instance is None:
        repo_root       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        checkpoint_path = os.path.join(repo_root, "best_model.pth")
        m = smp.UnetPlusPlus(
            encoder_name        = "efficientnet-b7",
            encoder_weights     = None,
            in_channels         = 3,
            classes             = NUM_CLASSES,
            decoder_attention_type = "scse",
        )
        if os.path.exists(checkpoint_path):
            ckpt  = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            m.load_state_dict(state)
            logging.info("Segmentation model loaded successfully.")
        else:
            logging.warning(f"best_model.pth not found at {checkpoint_path}.")
        m = m.to(_device)
        m.eval()
        _model_instance = m
    return _model_instance


def _run_segmentation(bgr: np.ndarray) -> dict:
    """
    Core segmentation pipeline (no gate):
      UNet++ segmentation → land cover masks + bounding boxes
    """
    model  = get_model()
    engine = BBoxEngine(
        model,
        device         = str(_device),
        min_confidence = MIN_CONFIDENCE,
        min_area_px    = MIN_AREA,
        nms_iou_thresh = NMS_IOU_THRESH,
        max_detections = MAX_DETECTIONS,
    )
    result = engine.predict_bgr(bgr)
    return {
        "detections":             result["detections"],
        "annotated_image_base64": result["annotated_b64"],
        "mask_image_base64":      result["mask_b64"],
        "image_width":            bgr.shape[1],
        "image_height":           bgr.shape[0],
        "summary":                result.get("summary", {}),
    }


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Full prediction pipeline (with satellite gate):
      1. Satellite gate check  → reject non-satellite images immediately
      2. UNet++ segmentation   → land cover masks + bounding boxes
    """
    pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil)
    bgr    = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    sat, score = is_satellite_image(pil)
    if not sat:
        logging.info(f"Gate rejected image  score={score}")
        return {
            "rejected":        True,
            "error":           "Not a satellite image",
            "satellite_score": score,
        }

    result = _run_segmentation(bgr)
    result["satellite_score"] = score
    return result


def predict_from_bytes_no_gate(image_bytes: bytes) -> dict:
    """
    Prediction pipeline WITHOUT satellite gate.
    Used for images fetched from ArcGIS (always satellite by definition).
    """
    pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil)
    bgr    = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return _run_segmentation(bgr)


# ==============================================================================
# 5. FLASK APP ROUTES & INITIALIZATION
# ==============================================================================
def create_app():
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.config["DATABASE"]   = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "app.db")
    )
    CORS(app, supports_credentials=True)

    @app.teardown_appcontext
    def close_db(exception=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    init_db(app)

    # ── Auth routes ──────────────────────────────────────────────────────────
    @app.route("/api/register", methods=["POST"])
    def register():
        data     = request.get_json(force=True) or {}
        name     = (data.get("name")     or "").strip()
        email    = (data.get("email")    or "").strip().lower()
        password =  data.get("password") or ""
        if not name or not email or not password:
            return jsonify({"error": "Name, email, and password are required."}), 400
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, hash_password(password), datetime.utcnow().isoformat()+"Z"),
            )
            db.commit()
        except sqlite3.IntegrityError:
            return jsonify({"error": "Email already registered."}), 400
        return jsonify({"message": "User created"}), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        data     = request.get_json(force=True) or {}
        email    = (data.get("email")    or "").strip().lower()
        password =  data.get("password") or ""
        if not email or not password:
            return jsonify({"error": "Email and password are required."}), 400
        db  = get_db()
        row = db.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return jsonify({"error": "Invalid email or password."}), 401
        token = create_session(row["id"])
        return jsonify({"token": token,
                        "user":  {"id": row["id"], "name": row["name"], "email": row["email"]}})

    @app.route("/api/logout", methods=["POST"])
    @token_required
    def logout():
        auth  = request.headers.get("Authorization", "")
        token = auth.split(" ", 1)[1] if " " in auth else None
        if token:
            db = get_db()
            db.execute("DELETE FROM sessions WHERE token = ?", (token,))
            db.commit()
        return jsonify({"message": "Logged out"})

    @app.route("/api/me", methods=["GET"])
    @token_required
    def me():
        return jsonify({"user": request.current_user})

    # ── Prediction routes ────────────────────────────────────────────────────
    @app.route("/api/predict", methods=["POST"])
    def predict_image():
        payload   = request.get_json(force=True) or {}
        image_b64 = payload.get("image_base64")
        if not image_b64:
            return jsonify({"error": "image_base64 is required"}), 400
        try:
            if image_b64.startswith("data:image"):
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return jsonify({"error": f"Invalid image_base64: {e}"}), 400
        try:
            result = predict_from_bytes(image_bytes)
        except Exception as e:
            logging.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        # Gate rejection returns 422 so frontend can show a specific message
        if result.get("rejected"):
            return jsonify(result), 422

        return jsonify(result)

    @app.route("/api/predict/coordinates", methods=["POST"])
    def predict_from_coordinates():
        payload  = request.get_json(force=True) or {}
        lat      = payload.get("lat")
        lon      = payload.get("lon")
        radius_m = payload.get("radius_m")
        size     = int(payload.get("size", 512))
        if lat is None or lon is None or radius_m is None:
            return jsonify({"error": "lat, lon, and radius_m are required"}), 400
        try:
            lat, lon, radius_m = float(lat), float(lon), float(radius_m)
        except Exception:
            return jsonify({"error": "lat, lon, and radius_m must be numbers"}), 400
        api_key = os.environ.get("ARCGIS_API_KEY")
        try:
            tile_bytes = fetch_arcgis_tile(lat=lat, lon=lon, radius_m=radius_m,
                                           size=size, api_key=api_key)
        except Exception as e:
            return jsonify({"error": f"Could not fetch satellite image: {e}"}), 500
        try:
            result = predict_from_bytes_no_gate(tile_bytes)
        except Exception as e:
            logging.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        return jsonify({
            "satellite_image_base64": base64.b64encode(tile_bytes).decode("ascii"),
            **result,
        })

    # ── Warmup both models at startup ────────────────────────────────────────
    try:
        get_model()
        logging.info("Segmentation model warmup completed")
    except Exception as err:
        logging.warning("Segmentation model warmup failed: %s", err)

    try:
        get_sat_classifier()                     # ← NEW warmup
        logging.info("Satellite gate v4 warmup completed")
    except Exception as err:
        logging.warning("Satellite gate warmup failed: %s", err)

    # ── Static file serving ──────────────────────────────────────────────────
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
