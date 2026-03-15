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
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp

from flask import Flask, send_from_directory, jsonify, request, g
from flask_cors import CORS

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# Controls the size of the images the model expects during inference.
# All input images are resized to 512x512 before being passed to the model.
IMAGE_SIZE = 512

# Class definitions for the Land Cover Segmentation model. 
# These MUST match the order used during training.
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

# Skip drawing bounding boxes for the `unknown` class
SKIP_CLASSES = {6}

# BGR Colors used by OpenCV to draw bounding boxes and masks.
CLASS_COLORS_BGR = {
    0: (0, 255, 255),    # urban_land -> yellow
    1: (0, 255, 0),      # agriculture -> green
    2: (0, 165, 255),    # rangeland -> orange
    3: (0, 100, 0),      # forest -> dark green
    4: (255, 0, 0),      # water -> blue
    5: (128, 128, 255),  # barren -> light red
    6: (80, 80, 80),     # unknown -> dark gray
}

# ------------------------------------------------------------------------------
# Bounding Box Filters
# ------------------------------------------------------------------------------
# We filter out model predictions that aren't confident enough or are too small.
# These can be overridden by environment variables for easy configuration.
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.55"))  # Minimum softmax probability to count as a detection
MIN_AREA = int(os.environ.get("MIN_AREA", "2000"))                # Ignore any detected blob smaller than 2000 pixels
NMS_IOU_THRESH = float(os.environ.get("NMS_IOU_THRESH", "0.5"))   # Overlap threshold for Non-Maximum Suppression (NMS)
MAX_DETECTIONS = int(os.environ.get("MAX_DETECTIONS", "25"))      # Limit the number of boxes we draw so the UI isn't cluttered


# ==============================================================================
# 1. DATABASE & AUTHENTICATION MODULE
# Handles user sessions, registration, and login using SQLite and Bcrypt.
# ==============================================================================
def get_db():
    """
    Get or create a SQLite database connection for the current Flask request.
    Stores the connection in the global `g` object to reuse it during the request.
    """
    if "db" not in g:
        from flask import current_app
        db_path = current_app.config["DATABASE"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # check_same_thread=False is needed because Flask might pass the connection across threads
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
        g.db = conn
    return g.db

def init_db(app):
    """
    Initialize the database tables for users and active sessions.
    Runs once when the Flask application starts up.
    """
    with app.app_context():
        db = get_db()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.commit()

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with a generated salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    """Checks if a plaintext password matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False

def create_session(user_id: int) -> str:
    """
    Creates a new login session for a user.
    Generates a secure random 32-byte token and stores it in the database.
    """
    token = secrets.token_urlsafe(32)
    created_at = datetime.utcnow().isoformat() + "Z"
    db = get_db()
    db.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, created_at),
    )
    db.commit()
    return token

def get_user_by_token(token: str):
    """Looks up a user in the database using their active session token."""
    if not token:
        return None
    db = get_db()
    row = db.execute(
        "SELECT u.id, u.name, u.email FROM users u JOIN sessions s ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    ).fetchone()
    return dict(row) if row else None

def token_required(fn):
    """
    Python Decorator to protect Flask API routes.
    Extracts the Bearer token from the Authorization header, validates the user,
    and blocks access (401 Unauthorized) if the token is missing or invalid.
    """
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
        else:
            token = None
        user = get_user_by_token(token)
        if user is None:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Attach the user to the request so the route handler can access it
        request.current_user = user
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ==============================================================================
# 2. SATELLITE TILE FETCHING MODULE
# Connects to ArcGIS MapServers to download satellite images dynamically.
# ==============================================================================
def _latlon_to_web_mercator(lat: float, lon: float):
    """
    Converts standard GPS coordinates (Latitude/Longitude) into Web Mercator (EPSG:3857).
    ArcGIS MapServers expect bounding boxes in this projection format to avoid distortion.
    """
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y

def fetch_arcgis_tile(lat: float, lon: float, radius_m: float, size: int = 512, api_key: str = None) -> bytes:
    """
    Downloads a square satellite image from ArcGIS centered on `lat`/`lon`.
    `radius_m` specifies how many meters out from the center the image should capture.
    Returns the raw PNG image bytes.
    """
    ARCGIS_EXPORT_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    
    # 1. Convert center point to web mercator
    center_x, center_y = _latlon_to_web_mercator(lat, lon)
    
    # 2. Calculate the bounding box edges by adding/subtracting the requested radius (meters)
    bbox = [
        center_x - radius_m,
        center_y - radius_m,
        center_x + radius_m,
        center_y + radius_m,
    ]

    # 3. Construct API parameters for ArcGIS
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": 3857,        # Input bounding box corresponds to Web Mercator projection
        "imageSR": 3857,       # Output image should be in Web Mercator projection
        "size": f"{size},{size}", # Image dimensions in pixels (512x512)
        "format": "png",       # Request image as PNG
        "f": "image",          # Return binary image data directly instead of JSON
        "dpi": 96,
    }
    if api_key:
        params["token"] = api_key

    # 4. Fetch the image tile
    resp = requests.get(ARCGIS_EXPORT_URL, params=params, timeout=60)
    resp.raise_for_status() # Throw exception if the request fails (404, 500 etc)
    return resp.content


# ==============================================================================
# 3. BOUNDING BOX ENGINE MODULE
# Post-processes the model predictions. 
# Converts the pixel-wise mask output of the U-Net model into exact bounding boxes.
# ==============================================================================
# Albumentations composed pipeline to normalize images to ImageNet stats and convert to PyTorch tensors.
# This MUST match what was used during model training.
_PREPROCESS = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class BBoxEngine:
    """
    Wraps the loaded deep learning segmentation model.
    Runs inference, exacts bounding boxes from the semantic masks, applies NMS filtering,
    and draws the final colored annotations onto the original image.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        min_confidence: float = 0.45,
        min_area_px: int = 400,
        nms_iou_thresh: float = 0.45,
        mask_alpha: float = 0.40,
        box_thickness: int = 2,
        font_scale: float = 0.55,
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
        self.model.eval() # Ensure model is in evaluation (not training) mode

    def predict_bgr(self, bgr: np.ndarray) -> dict:
        """
        Main pipeline method: Takes an un-preprocessed BGR OpenCV image and returns complete predictions.
        """
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Run the image through the neural network to get masks and probabilities
        class_mask, probs = self._run_model(rgb, orig_h, orig_w)

        # 2. Group adjacent pixels of the same class into Contours -> Bounding Boxes
        detections = self._mask_to_bboxes(class_mask, probs)
        
        # 3. Suppress overlapping bounding boxes using Non-Maximum Suppression (NMS)
        detections = self._nms(detections)

        # 4. Sort and filter the top Detections
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        if self.max_detections and len(detections) > self.max_detections:
            # Sort by Area + Confidence combined if we need to drop boxes, keeping the largest/most confident
            detections = sorted(
                detections,
                key=lambda d: (d.get("area", 0), d.get("confidence", 0)),
                reverse=True,
            )[:self.max_detections]

        # 5. Drawing: Overlay semantic masks and bounding boxes over the original image
        mask_bgr       = self._colorize_mask(class_mask, orig_h, orig_w)
        annotated_bgr  = self._draw_all(bgr, mask_bgr, detections)

        # 6. Encode drawn images as base64 JPEG strings so they can be JSON-serialized for the web frontend
        annotated_b64 = self._encode_b64(annotated_bgr)
        mask_b64      = self._encode_b64(mask_bgr)

        # 7. Generate a simple count summary for all found objects
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
    def _run_model(self, rgb: np.ndarray, orig_h: int, orig_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the semantic segmentation model on the image."""
        # Resize image cleanly, apply normalization, and move it to the GPU
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor  = _PREPROCESS(image=resized)["image"].unsqueeze(0).to(self.device)

        # Automatic Mixed Precision (AMP) makes execution much faster on CUDA devices
        use_amp = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = self.model(tensor)

        # The model outputs a tensor of size 512x512. We up-scale it back to the ORIGINAL dimensions of the user image.
        logits_up = F.interpolate(logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        
        # Softmax determines probabilities (0-1) for each pixel being a given class
        probs = F.softmax(logits_up, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        
        # Argmax takes the highest probability class identifier for each pixel (creating a single 2D grid of IDs)
        class_mask = probs.argmax(axis=0).astype(np.int32)            # (H, W)
        return class_mask, probs

    def _mask_to_bboxes(self, class_mask: np.ndarray, probs: np.ndarray) -> List[Dict]:
        """Converts pixel clusters from the class_mask into actual rectangle Bounding Boxes."""
        detections = []
        for cid in range(NUM_CLASSES):
            if cid in SKIP_CLASSES:
                continue

            # Create a 2D binary grid indicating ONLY pixels belonging to THIS class (cid)
            binary = (class_mask == cid).astype(np.uint8)
            if binary.sum() < self.min_area:
                continue

            # Morphological Close: Fills in small gaps/holes so a single field/building isn't fragmented into 5 pieces
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Connected Components finds isolated islands/blobs of pixels that touch each other
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)

            for lid in range(1, n_labels): # Ignore label 0 (which is the background)
                area = int(stats[lid, cv2.CC_STAT_AREA])
                
                # Ignore blobs that are too small
                if area < self.min_area:
                    continue

                # Basic XY position and dimensions supplied by ConnectedComponents
                x = int(stats[lid, cv2.CC_STAT_LEFT])
                y = int(stats[lid, cv2.CC_STAT_TOP])
                w = int(stats[lid, cv2.CC_STAT_WIDTH])
                h = int(stats[lid, cv2.CC_STAT_HEIGHT])

                # Calculate average AI confidence inside this specific Blob/Component isolation
                comp_pixels = (labels == lid)
                mean_conf   = float(probs[cid][comp_pixels].mean())
                if mean_conf < self.min_conf:
                    continue

                # Find the exact tightest contour for this blob, which is often a better bounding rect
                # than the one given by Connected Components.
                comp_mask = comp_pixels.astype(np.uint8)
                cnts, _   = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Non-Maximum Suppression (NMS).
        If two bounding boxes for the SAME class overlap heavily (exceeding nms_thresh),
        drop the one with the lower confidence score to avoid duplicated boxes drawing over each other.
        """
        if not detections:
            return []
        out = []
        for cid in set(d["class_id"] for d in detections):
            dets = sorted([d for d in detections if d["class_id"] == cid], key=lambda d: d["confidence"], reverse=True)
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                # Keep only detections that DO NOT overlap the current 'best' box
                dets = [d for d in dets if self._iou(best["bbox"], d["bbox"]) < self.nms_thresh]
            out.extend(keep)
        return out

    def _iou(self, box_a: List[int], box_b: List[int]) -> float:
        """Calculates Intersection Over Union (IoU) ratio between two boxes to measure their overlap."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _colorize_mask(self, class_mask: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Generates a full RGB image representing the segmentation mask colors."""
        color_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cid, color in CLASS_COLORS_BGR.items():
            color_img[class_mask == cid] = color
        return color_img

    def _draw_all(self, bgr: np.ndarray, mask_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draws the transparent mask overlay AND string-labeled bounding boxes on top of the original image."""
        # Blend the original satellite image with the colorized pixel mask
        canvas = cv2.addWeighted(bgr, 1 - self.mask_alpha, mask_bgr, self.mask_alpha, 0)
        
        # Iterate over all final filtered detections and draw rectangular boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cid   = det["class_id"]
            name  = det["class_name"]
            conf  = det["confidence"]
            color = CLASS_COLORS_BGR[cid]

            # Draw outer rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Format text label (e.g., 'forest 0.98')
            label = f"{name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            pad = 4
            chip_y1 = max(y1 - th - 2 * pad, 0)
            chip_y2 = chip_y1 + th + 2 * pad
            chip_x2 = x1 + tw + 2 * pad

            # Draw a solid background box behind the text label so it's readable
            cv2.rectangle(canvas, (x1, chip_y1), (chip_x2, chip_y2), color, -1)
            
            # Use black text for bright color boxes, white text for dark ones
            brightness = int(color[0])*0.114 + int(color[1])*0.587 + int(color[2])*0.299
            text_color = (0, 0, 0) if brightness > 130 else (255, 255, 255)
            
            # Write final text on the image
            cv2.putText(
                canvas, label,
                (x1 + pad, chip_y2 - pad - bl // 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                text_color, 1, cv2.LINE_AA,
            )
        return canvas

    def _encode_b64(self, bgr: np.ndarray, quality: int = 90) -> str:
        """Compresses the OpenCV BGR image into a memory buffer and encodes it as Base64."""
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


# ==============================================================================
# 4. MODEL LOADER AND PREDICTION PROCESSING
# Efficiently manages the lifecycle of the PyTorch neural network
# ==============================================================================
_model_instance = None # Singleton instance, so the giant model is only loaded ONCE
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    """
    Lazy-loads the PyTorch model.
    It provisions the UnetPlusPlus architecture and maps the saved `best_model.pth` weights.
    By making this a singleton globally, we avoid blowing up GPU memory.
    """
    global _model_instance
    if _model_instance is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        checkpoint_path = os.path.join(repo_root, "best_model.pth")
        
        # Instantiate the architecture that matches the trained weights file
        m = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            decoder_attention_type="scse",
        )
        
        # Hydrate the model architecture with our trained checkpoint
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            m.load_state_dict(state)
            logging.info("Model checkpoint loaded successfully.")
        else:
            logging.warning(f"Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")
        
        m = m.to(_device)
        m.eval()
        _model_instance = m
    return _model_instance

def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Main orchestration function for image prediction.
    Takes physical image bits, converts them using PIL to cv2 arrays, initializes
    the BBoxEngine, and triggers the inference and drawing pipeline.
    """
    # Parse incoming bytes into PIL, then convert to OpenCV compatible BGR array
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Instantiate the Engine with our loaded model and current configuration
    model = get_model()
    engine = BBoxEngine(
        model,
        device=str(_device),
        min_confidence=MIN_CONFIDENCE,
        min_area_px=MIN_AREA,
        nms_iou_thresh=NMS_IOU_THRESH,
        max_detections=MAX_DETECTIONS
    )
    
    # Run prediction and BBox drawing
    result = engine.predict_bgr(bgr)

    # Return structured data for the Flask Endpoint to serialize into JSON
    return {
        "detections": result["detections"],
        "annotated_image_base64": result["annotated_b64"],
        "mask_image_base64": result["mask_b64"],
        "image_width": bgr.shape[1],
        "image_height": bgr.shape[0],
        "summary": result.get("summary", {})
    }


# ==============================================================================
# 5. FLASK APP ROUTES & INITIALIZATION
# Starts the web server and hooks up HTTP Endpoints to our python functions.
# ==============================================================================
def create_app():
    # Setup paths mapping to our Static files (.html, .css, .js)
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.config["DATABASE"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.db"))

    CORS(app, supports_credentials=True)

    # Ensure the DB connection closes cleanly after every user request
    @app.teardown_appcontext
    def close_db(exception=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # Pre-configure tables
    init_db(app)

    # ---------------------------------------------
    # Auth Endpoints (Register, Login, Logout)
    # ---------------------------------------------
    @app.route("/api/register", methods=["POST"])
    def register():
        data = request.get_json(force=True) or {}
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not name or not email or not password:
            return jsonify({"error": "Name, email, and password are required."}), 400

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, hash_password(password), datetime.utcnow().isoformat() + "Z"),
            )
            db.commit()
        except sqlite3.IntegrityError:
            return jsonify({"error": "Email already registered."}), 400
        return jsonify({"message": "User created"}), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        data = request.get_json(force=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        if not email or not password:
            return jsonify({"error": "Email and password are required."}), 400

        db = get_db()
        row = db.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        
        # Verify the hash matches what was provided
        if row is None or not verify_password(password, row["password_hash"]):
            return jsonify({"error": "Invalid email or password."}), 401

        # Hand back the session token for future requests
        token = create_session(row["id"])
        return jsonify({"token": token, "user": {"id": row["id"], "name": row["name"], "email": row["email"]}})

    @app.route("/api/logout", methods=["POST"])
    @token_required
    def logout():
        auth = request.headers.get("Authorization", "")
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

    # ---------------------------------------------
    # AI Prediction Endpoints 
    # ---------------------------------------------
    @app.route("/api/predict", methods=["POST"])
    def predict_image():
        """Accepts a direct Base64 encoded Image string, analyzes it, and returns annotated images and data."""
        payload = request.get_json(force=True) or {}
        image_b64 = payload.get("image_base64")
        if not image_b64:
            return jsonify({"error": "image_base64 is required"}), 400

        try:
            # Strip the meta-tag if present (from UI canvas encoding)
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
        return jsonify(result)

    @app.route("/api/predict/coordinates", methods=["POST"])
    def predict_from_coordinates():
        """Accepts map coordinates + radius, downloads satellite image from ArcGIS, and then analyzes it."""
        payload = request.get_json(force=True) or {}
        lat, lon = payload.get("lat"), payload.get("lon")
        radius_m = payload.get("radius_m")
        size = int(payload.get("size", 512)) # Tile size defaults to 512px

        if lat is None or lon is None or radius_m is None:
            return jsonify({"error": "lat, lon, and radius_m are required"}), 400
        try:
            lat, lon, radius_m = float(lat), float(lon), float(radius_m)
        except Exception:
            return jsonify({"error": "lat, lon, and radius_m must be numbers"}), 400

        # Optional: Load an API key from system environment variables for better limits
        api_key = os.environ.get("ARCGIS_API_KEY")
        
        # 1. Fetch satellite Image
        try:
            tile_bytes = fetch_arcgis_tile(lat=lat, lon=lon, radius_m=radius_m, size=size, api_key=api_key)
        except Exception as e:
            return jsonify({"error": f"Could not fetch satellite image: {e}"}), 500

        # 2. Run Image through AI Prediction Service
        try:
            result = predict_from_bytes(tile_bytes)
        except Exception as e:
             logging.exception("Prediction failed")
             return jsonify({"error": f"Prediction failed: {e}"}), 500

        # Return results along with the original satellite byte stream so the frontend can preview it
        return jsonify({
            "satellite_image_base64": base64.b64encode(tile_bytes).decode("ascii"),
            **result,
        })

    # Start loading the AI Model in the background during app startup to prevent the first request from being slow
    try:
        get_model()
        logging.info("Model warmup completed")
    except Exception as err:
        logging.warning("Model warmup failed: %s", err)

    # ---------------------------------------------
    # Fallback Route: Serve HTML/CSS/JS Assets
    # ---------------------------------------------
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        # Serve exact file if it exists, otherwise fall through to `index.html` (for Single Page App routing)
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)