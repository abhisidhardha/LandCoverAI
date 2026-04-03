# evaluate.py — Full evaluation script for LandCoverAI (UNet++ EfficientNet-B7)
# Usage:
#   python evaluate.py --split train    ← full metrics (has masks)
#   python evaluate.py --split val      ← inference + visual overlays only
#   python evaluate.py --split test     ← inference + visual overlays only
# ==============================================================================
import os, sys, math, time, json, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score,
    average_precision_score, roc_curve,
)
from sklearn.calibration import calibration_curve

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMAGE_SIZE      = 512
NUM_CLASSES     = 7
CLASS_NAMES     = [
    "urban_land",       # 0
    "agriculture_land", # 1
    "rangeland",        # 2
    "forest_land",      # 3
    "water",            # 4
    "barren_land",      # 5
    "unknown",          # 6
]
SKIP_CLASSES    = {6}
EVAL_CLASSES    = [c for c in range(NUM_CLASSES) if c not in SKIP_CLASSES]
PROB_SAMPLE_MAX = 500_000   # pixels kept for ROC/ECE/calibration

# ── DeepGlobe colour palette (from class_dict.csv) ────────────────────────────
_MASK_RGB_TO_CLASS = {
    (  0, 255, 255): 0,   # urban_land
    (255, 255,   0): 1,   # agriculture_land
    (255,   0, 255): 2,   # rangeland
    (  0, 255,   0): 3,   # forest_land
    (  0,   0, 255): 4,   # water
    (255, 255, 255): 5,   # barren_land
    (  0,   0,   0): 6,   # unknown
}

# BGR colours for OpenCV overlays (matches app.py)
CLASS_COLORS_BGR = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (0, 100, 0),
    4: (255, 0, 0),
    5: (128, 128, 255),
    6: (80, 80, 80),
}

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", os.path.join(os.path.dirname(__file__), "dataset"))
CHECKPOINT   = os.environ.get(
    "CHECKPOINT",   os.path.join(os.path.dirname(__file__),"best_model.pth"))
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "eval_output")
BATCH_SIZE   = int(os.environ.get("EVAL_BATCH_SIZE", "4"))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "evaluate.log")),
    ],
)
log = logging.getLogger(__name__)

# ==============================================================================
# 1.  TRANSFORMS
# ==============================================================================
VAL_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==============================================================================
# 2.  MASK DECODER
# ==============================================================================
# Build a fast lookup table: encode RGB → class index using a single numpy op
_LUT = np.full((256, 256, 256), 6, dtype=np.uint8)   # default = unknown
for (r, g, b), cls_id in _MASK_RGB_TO_CLASS.items():
    _LUT[r, g, b] = cls_id


def decode_color_mask(mask_path: Path) -> np.ndarray:
    """Convert DeepGlobe RGB colour mask → (H,W) class-index array (fast LUT)."""
    rgb = np.array(Image.open(mask_path).convert("RGB"))   # (H,W,3) uint8
    return _LUT[rgb[..., 0], rgb[..., 1], rgb[..., 2]].astype(np.int64)

# ==============================================================================
# 3.  DATASETS
# ==============================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def detect_split_type(root: str, split: str) -> str:
    split_dir = Path(root) / split
    return "supervised" if any(split_dir.glob("*_mask.png")) else "inference"


class SupervisedDataset(torch.utils.data.Dataset):
    """train/ — *_sat.jpg paired with *_mask.png in same folder."""

    def __init__(self, root: str, split: str):
        split_dir  = Path(root) / split
        sat_paths  = sorted(split_dir.glob("*_sat.jpg"))
        assert len(sat_paths) > 0, f"No *_sat.jpg in {split_dir}"

        self.pairs = []
        skipped    = 0
        for sat in sat_paths:
            stem = sat.name.replace("_sat.jpg", "")
            mask = split_dir / f"{stem}_mask.png"
            if mask.exists():
                self.pairs.append((sat, mask))
            else:
                skipped += 1

        if skipped:
            log.warning(f"Skipped {skipped} sat images with no matching mask.")
        assert len(self.pairs) > 0, f"No valid pairs in {split_dir}"
        log.info(f"SupervisedDataset [{split}]: {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sat_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(sat_path).convert("RGB"))
        mask = decode_color_mask(mask_path)                 # (H,W) int64
        t    = VAL_TRANSFORM(image=img, mask=mask.astype(np.int32))
        return t["image"], t["mask"].long(), str(sat_path)


class InferenceDataset(torch.utils.data.Dataset):
    """val/ or test/ — only *_sat.jpg, no masks."""

    def __init__(self, root: str, split: str):
        split_dir  = Path(root) / split
        self.imgs  = sorted(split_dir.glob("*_sat.jpg"))
        if not self.imgs:
            log.warning("No *_sat.jpg found — using all image files.")
            self.imgs = sorted(
                p for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        assert len(self.imgs) > 0, f"No images in {split_dir}"
        log.info(f"InferenceDataset [{split}]: {len(self.imgs)} images.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img  = np.array(Image.open(path).convert("RGB"))
        orig = img.copy()
        t    = VAL_TRANSFORM(image=img)
        return t["image"], orig, str(path)

# ==============================================================================
# 4.  MODEL
# ==============================================================================
def load_model(ckpt_path: str) -> nn.Module:
    log.info("Loading UNet++ (EfficientNet-B7)…")
    model = smp.UnetPlusPlus(
        encoder_name           = "efficientnet-b7",
        encoder_weights        = None,
        in_channels            = 3,
        classes                = NUM_CLASSES,
        decoder_attention_type = "scse",
    )
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        log.info(f"✔ Checkpoint loaded: {ckpt_path}")
    else:
        log.warning(f"✘ Checkpoint NOT found at {ckpt_path} — random weights!")
    return model.to(DEVICE).eval()


def model_complexity(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info  = {
        "Total_Params_M":     round(total / 1e6, 2),
        "Trainable_Params_M": round(train / 1e6, 2),
    }
    try:
        from thop import profile
        dummy        = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        flops, _     = profile(model, inputs=(dummy,), verbose=False)
        info["GFLOPs"] = round(flops / 1e9, 2)
    except ImportError:
        log.info("pip install thop  →  enables GFLOPs count")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        info["GPU_Mem_GB"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
    return info

# ==============================================================================
# 5.  INFERENCE LOOPS
# ==============================================================================
@torch.inference_mode()
def run_supervised(model: nn.Module, loader: torch.utils.data.DataLoader):
    """
    Streaming inference — accumulates confusion matrix directly.
    Never stores all-pixel probability arrays to avoid OOM.
    """
    cm_accum        = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    prob_reservoir  = []
    gt_reservoir    = []
    reservoir_count = 0
    t_total         = 0.0
    n_imgs          = 0

    pbar = tqdm(
        loader,
        desc  = "  Inference",
        unit  = "batch",
        ncols = 90,
        colour= "cyan",
        dynamic_ncols=True,
    )

    for imgs, masks, _ in pbar:
        imgs = imgs.to(DEVICE)

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(imgs)
        t_total += time.perf_counter() - t0
        n_imgs  += imgs.size(0)

        logits_up   = F.interpolate(
            logits.float(), size=masks.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        probs_batch = F.softmax(logits_up, dim=1).cpu().numpy()   # (B,C,H,W)
        preds_batch = probs_batch.argmax(axis=1)                   # (B,H,W)
        masks_np    = masks.numpy()                                 # (B,H,W)

        for b in range(imgs.size(0)):
            p_flat = preds_batch[b].ravel()
            g_flat = masks_np[b].ravel()

            # ── Streaming confusion matrix update ────────────────────────────
            np.add.at(cm_accum, (g_flat, p_flat), 1)

            # ── Reservoir sampling for probs ─────────────────────────────────
            if reservoir_count < PROB_SAMPLE_MAX:
                pr        = probs_batch[b].reshape(NUM_CLASSES, -1).T
                remaining = PROB_SAMPLE_MAX - reservoir_count
                take      = min(len(p_flat), remaining)
                idx       = np.random.choice(len(p_flat), take, replace=False)
                prob_reservoir.append(pr[idx].astype(np.float32))
                gt_reservoir.append(g_flat[idx].astype(np.int8))
                reservoir_count += take

        # Live stats in progress bar
        cur_acc = float(np.diag(cm_accum).sum() / max(cm_accum.sum(), 1))
        pbar.set_postfix({
            "imgs"    : n_imgs,
            "acc"     : f"{cur_acc:.3f}",
            "ms/img"  : f"{t_total/n_imgs*1000:.0f}",
        })

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    perf = {
        "latency_ms": round(t_total / n_imgs * 1000, 2),
        "fps"        : round(n_imgs / t_total, 3),
        "n_images"   : int(n_imgs),
    }
    log.info(
        f"Inference complete — {n_imgs} imgs | "
        f"{perf['latency_ms']} ms/img | {perf['fps']} FPS"
    )

    probs_sample = np.concatenate(prob_reservoir, axis=0)
    gts_sample   = np.concatenate(gt_reservoir,   axis=0).astype(np.int64)
    return cm_accum, probs_sample, gts_sample, perf


@torch.inference_mode()
def run_inference_only(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    out_dir: str,
    split: str,
) -> dict:
    """Inference-only for splits without masks — saves visual overlays."""
    vis_dir = os.path.join(out_dir, f"predictions_{split}")
    os.makedirs(vis_dir, exist_ok=True)

    rows    = []
    t_total = 0.0
    n_imgs  = 0

    pbar = tqdm(
        loader,
        desc  = "  Inference",
        unit  = "batch",
        ncols = 90,
        colour= "green",
        dynamic_ncols=True,
    )

    for imgs_t, origs, paths in pbar:
        imgs_t = imgs_t.to(DEVICE)

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(imgs_t)
        t_total += time.perf_counter() - t0
        n_imgs  += imgs_t.size(0)

        orig_h, orig_w = origs[0].shape[:2] if isinstance(origs[0], np.ndarray) \
                         else (origs[0].shape[0], origs[0].shape[1])

        logits_up = F.interpolate(
            logits.float(), size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )
        probs  = F.softmax(logits_up, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        confs  = probs.max(axis=1)

        for b in range(imgs_t.size(0)):
            stem     = Path(paths[b]).stem
            orig_np  = origs[b] if isinstance(origs[b], np.ndarray) else np.array(origs[b])
            orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
            pred_map = preds[b]

            # Colour overlay
            color_mask = np.zeros_like(orig_bgr)
            for cid, color in CLASS_COLORS_BGR.items():
                color_mask[pred_map == cid] = color
            overlay = cv2.addWeighted(orig_bgr, 0.6, color_mask, 0.4, 0)

            # Confidence heatmap
            conf_norm = (confs[b] * 255).astype(np.uint8)
            conf_heat = cv2.applyColorMap(conf_norm, cv2.COLORMAP_TURBO)

            # Legend strip
            lh = 30 * NUM_CLASSES
            legend = np.ones((lh, orig_w, 3), dtype=np.uint8) * 240
            for i, (cid, cname) in enumerate(zip(range(NUM_CLASSES), CLASS_NAMES)):
                y1, y2 = i * 30, (i + 1) * 30
                cv2.rectangle(legend, (5, y1+4), (35, y2-4), CLASS_COLORS_BGR[cid], -1)
                pct = float((pred_map == cid).mean() * 100)
                cv2.putText(legend, f"{cname}  {pct:.1f}%",
                            (45, y2-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (30, 30, 30), 1, cv2.LINE_AA)

            panel = cv2.hconcat([orig_bgr, overlay, conf_heat])
            panel = cv2.vconcat([panel,
                                 cv2.resize(legend, (panel.shape[1], lh))])
            cv2.imwrite(
                os.path.join(vis_dir, f"{stem}_pred.jpg"),
                panel,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            row = {"filename": Path(paths[b]).name}
            for cid, cname in enumerate(CLASS_NAMES):
                m = pred_map == cid
                row[f"{cname}_pct"]       = round(float(m.mean() * 100), 2)
                row[f"{cname}_mean_conf"] = round(
                    float(probs[b][cid][m].mean()) if m.any() else 0.0, 4)
            rows.append(row)

        pbar.set_postfix({"imgs": n_imgs, "ms/img": f"{t_total/n_imgs*1000:.0f}"})

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    perf = {
        "latency_ms": round(t_total / n_imgs * 1000, 2) if n_imgs else 0,
        "fps"        : round(n_imgs / t_total, 3)        if t_total else 0,
        "n_images"   : int(n_imgs),
    }

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(out_dir, f"inference_{split}_{ts}.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    log.info(f"Per-image distribution CSV: {csv}")
    log.info(f"Overlay images → {vis_dir}/")
    return perf

# ==============================================================================
# 6.  METRICS ENGINE
# ==============================================================================
def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)


def compute_all_metrics(
    cm: np.ndarray,
    probs_sample: np.ndarray,
    gts_sample: np.ndarray,
    perf: dict,
) -> tuple:

    steps = [
        "Base stats (TP/FP/FN/TN)",
        "Pixel accuracy",
        "IoU / FWIoU",
        "Dice / F1",
        "Precision / Recall / Specificity",
        "Cohen Kappa",
        "MCC",
        "ROC-AUC & mAP",
        "ECE (calibration)",
        "Per-class DataFrame",
    ]
    pbar = tqdm(steps, desc="  Metrics", unit="step",
                ncols=90, colour="yellow", dynamic_ncols=True)

    results = {}

    # ── Base stats ─────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › Base stats"); next(iter([pbar.update(0)]))
    tp   = np.diag(cm)
    fp   = cm.sum(axis=0) - tp
    fn   = cm.sum(axis=1) - tp
    tn   = cm.sum() - (tp + fp + fn)
    freq = cm.sum(axis=1) / cm.sum()
    pbar.update(1)

    iou_per_class  = safe_div(tp, tp + fp + fn)
    dice_per_class = safe_div(2*tp, 2*tp + fp + fn)
    prec_per_class = safe_div(tp, tp + fp)
    rec_per_class  = safe_div(tp, tp + fn)
    acc_per_class  = safe_div(tp, cm.sum(axis=1))
    spec_per_class = safe_div(tn, tn + fp)

    # ── Pixel accuracy ──────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › Pixel accuracy")
    results["Overall_Pixel_Accuracy"] = float(np.diag(cm).sum() / cm.sum())
    results["Mean_Class_Accuracy"]    = float(acc_per_class[EVAL_CLASSES].mean())
    results["FW_Pixel_Accuracy"]      = float((freq * acc_per_class).sum())
    pbar.update(1)

    # ── IoU ─────────────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › IoU / FWIoU")
    results["mIoU"]  = float(iou_per_class[EVAL_CLASSES].mean())
    results["FWIoU"] = float((freq * iou_per_class).sum())
    pbar.update(1)

    # ── Dice / F1 ───────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › Dice / F1")
    results["Mean_Dice"] = float(dice_per_class[EVAL_CLASSES].mean())
    results["Macro_F1"]  = results["Mean_Dice"]
    tp_m = tp[EVAL_CLASSES].sum()
    fp_m = fp[EVAL_CLASSES].sum()
    fn_m = fn[EVAL_CLASSES].sum()
    results["Micro_F1"] = float(safe_div(2*tp_m, 2*tp_m + fp_m + fn_m))
    pbar.update(1)

    # ── Precision / Recall / Specificity ────────────────────────────────────────
    pbar.set_description("  Metrics › Precision / Recall / Specificity")
    results["Macro_Precision"]  = float(prec_per_class[EVAL_CLASSES].mean())
    results["Macro_Recall"]     = float(rec_per_class[EVAL_CLASSES].mean())
    results["Mean_Specificity"] = float(spec_per_class[EVAL_CLASSES].mean())
    pbar.update(1)

    # ── Cohen Kappa ─────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › Cohen Kappa")
    kappa_gts, kappa_preds = [], []
    for gt_c in tqdm(EVAL_CLASSES, desc="    Kappa rows",
                     leave=False, ncols=80, colour="magenta"):
        for pred_c in EVAL_CLASSES:
            count = int(cm[gt_c, pred_c])
            if count > 0:
                kappa_gts.extend([gt_c]    * count)
                kappa_preds.extend([pred_c] * count)

    MAX_KAPPA = 2_000_000
    if len(kappa_gts) > MAX_KAPPA:
        idx         = np.random.choice(len(kappa_gts), MAX_KAPPA, replace=False)
        kappa_gts   = np.array(kappa_gts)[idx]
        kappa_preds = np.array(kappa_preds)[idx]

    try:
        results["Cohen_Kappa"] = float(cohen_kappa_score(kappa_gts, kappa_preds))
    except Exception:
        results["Cohen_Kappa"] = 0.0
    pbar.update(1)

    # ── MCC ─────────────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › MCC")
    mcc_vals = []
    for c in EVAL_CLASSES:
        tp_c  = int(cm[c, c])
        fp_c  = int(cm[:, c].sum() - tp_c)
        fn_c  = int(cm[c, :].sum() - tp_c)
        tn_c  = int(cm.sum() - tp_c - fp_c - fn_c)
        denom = math.sqrt(
            max((tp_c+fp_c)*(tp_c+fn_c)*(tn_c+fp_c)*(tn_c+fn_c), 1e-9))
        mcc_vals.append((tp_c*tn_c - fp_c*fn_c) / denom)
    results["Mean_MCC"] = float(np.mean(mcc_vals))
    pbar.update(1)

    # ── ROC-AUC & mAP ───────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › ROC-AUC & mAP")
    try:
        oh    = np.zeros((len(gts_sample), NUM_CLASSES), dtype=np.float32)
        oh[np.arange(len(gts_sample)), gts_sample] = 1
        auc_v, ap_v = [], []
        for c in tqdm(EVAL_CLASSES, desc="    AUC/AP per class",
                      leave=False, ncols=80, colour="magenta"):
            if oh[:, c].sum() == 0:
                continue
            auc_v.append(roc_auc_score(oh[:, c], probs_sample[:, c]))
            ap_v.append(average_precision_score(oh[:, c], probs_sample[:, c]))
        results["Macro_ROC_AUC"] = float(np.mean(auc_v))
        results["Macro_mAP"]     = float(np.mean(ap_v))
    except Exception as e:
        log.warning(f"ROC/AP skipped: {e}")
    pbar.update(1)

    # ── ECE ─────────────────────────────────────────────────────────────────────
    pbar.set_description("  Metrics › ECE")
    try:
        conf = probs_sample.max(axis=1)
        corr = (probs_sample.argmax(axis=1) == gts_sample).astype(float)
        bins = np.linspace(0, 1, 16)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.sum():
                ece += abs(corr[m].mean() - conf[m].mean()) * m.mean()
        results["ECE"] = float(ece)
    except Exception as e:
        log.warning(f"ECE skipped: {e}")
    pbar.update(1)

    # ── Per-class DataFrame ─────────────────────────────────────────────────────
    pbar.set_description("  Metrics › Per-class DataFrame")
    per_class = pd.DataFrame({
        "Class":       [CLASS_NAMES[c] for c in range(NUM_CLASSES)],
        "IoU":         iou_per_class.round(4).tolist(),
        "Dice":        dice_per_class.round(4).tolist(),
        "Precision":   prec_per_class.round(4).tolist(),
        "Recall":      rec_per_class.round(4).tolist(),
        "F1":          dice_per_class.round(4).tolist(),
        "Accuracy":    acc_per_class.round(4).tolist(),
        "Specificity": spec_per_class.round(4).tolist(),
        "TP": tp.tolist(), "FP": fp.tolist(),
        "FN": fn.tolist(), "TN": tn.tolist(),
    })
    pbar.update(1)
    pbar.close()

    results.update(perf)
    return results, per_class, cm

# ==============================================================================
# 7.  VISUALIZATIONS
# ==============================================================================
def _save_plots(summary, per_class_df, cm, probs_sample, gts_sample,
                out_dir: str, split: str):

    plot_tasks = [
        "Confusion matrix",
        "Per-class bar chart",
        "Calibration / reliability diagram",
        "ROC curves",
        "Class distribution pie chart",
    ]
    pbar = tqdm(plot_tasks, desc="  Plots", unit="plot",
                ncols=90, colour="blue", dynamic_ncols=True)

    # ── 1. Confusion matrix ────────────────────────────────────────────────────
    pbar.set_description("  Plots › Confusion matrix")
    fig, ax = plt.subplots(figsize=(9, 7))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.4, vmin=0, vmax=1)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix — {split}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{split}.png"), dpi=180)
    plt.close()
    pbar.update(1)

    # ── 2. Per-class bar chart ─────────────────────────────────────────────────
    pbar.set_description("  Plots › Per-class bar chart")
    metrics = ["IoU", "Dice", "Precision", "Recall", "F1"]
    df      = per_class_df[per_class_df["Class"] != "unknown"].reset_index(drop=True)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(df)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4), sharey=True)
    for ax, m in zip(axes, metrics):
        bars = ax.barh(df["Class"], df[m], color=colors, edgecolor="k", lw=0.4)
        ax.set_xlabel(m, fontsize=11); ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, df[m]):
            ax.text(min(v + 0.01, 0.96), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle(f"Per-class Segmentation Metrics — {split}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"per_class_metrics_{split}.png"),
                dpi=180, bbox_inches="tight")
    plt.close()
    pbar.update(1)

    # ── 3. Calibration / Reliability diagram ──────────────────────────────────
    pbar.set_description("  Plots › Calibration diagram")
    try:
        conf = probs_sample.max(axis=1)
        acc  = (probs_sample.argmax(axis=1) == gts_sample).astype(float)
        fp_c, mp_c = calibration_curve(acc, conf, n_bins=15, strategy="uniform")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mp_c, fp_c, "o-", label="Model", color="steelblue")
        ax.plot([0, 1], [0, 1], "--k", label="Perfect calibration", lw=1)
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Fraction correct")
        ax.set_title(f"Reliability Diagram — {split}")
        ece_val = summary.get("ECE", "?")
        ax.text(0.05, 0.92, f"ECE = {ece_val:.4f}" if isinstance(ece_val, float) else "",
                transform=ax.transAxes, fontsize=10, color="red")
        ax.legend(); ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"calibration_{split}.png"), dpi=160)
        plt.close()
    except Exception as e:
        log.warning(f"Calibration plot skipped: {e}")
    pbar.update(1)

    # ── 4. ROC curves ──────────────────────────────────────────────────────────
    pbar.set_description("  Plots › ROC curves")
    try:
        oh  = np.zeros((len(gts_sample), NUM_CLASSES), dtype=np.float32)
        oh[np.arange(len(gts_sample)), gts_sample] = 1
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in EVAL_CLASSES:
            if oh[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(oh[:, c], probs_sample[:, c])
            auc_val     = roc_auc_score(oh[:, c], probs_sample[:, c])
            ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]}  (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "--k", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curves (One-vs-Rest) — {split}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_curves_{split}.png"), dpi=160)
        plt.close()
    except Exception as e:
        log.warning(f"ROC plot skipped: {e}")
    pbar.update(1)

    # ── 5. Class distribution pie chart ───────────────────────────────────────
    pbar.set_description("  Plots › Class distribution pie")
    try:
        class_totals = cm.sum(axis=1)
        labels_pie   = [f"{n}\n{v:,}" for n, v in zip(CLASS_NAMES, class_totals)]
        colors_pie   = [
            tuple(c / 255 for c in CLASS_COLORS_BGR[i][::-1]) + (0.85,)
            for i in range(NUM_CLASSES)
        ]
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            class_totals, labels=labels_pie, colors=colors_pie,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(f"Ground Truth Class Distribution — {split}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"class_distribution_{split}.png"), dpi=160)
        plt.close()
    except Exception as e:
        log.warning(f"Pie chart skipped: {e}")
    pbar.update(1)

    pbar.close()
    log.info(f"All plots saved → {out_dir}/")

# ==============================================================================
# 8.  SAVE OUTPUTS
# ==============================================================================
def save_outputs(summary: dict, per_class_df: pd.DataFrame,
                 out_dir: str, split: str):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = [
        f"summary_{split}_{ts}.csv",
        f"per_class_{split}_{ts}.csv",
        f"metrics_{split}_{ts}.json",
    ]
    pbar = tqdm(tasks, desc="  Saving CSVs/JSON", unit="file",
                ncols=90, colour="white", dynamic_ncols=True)

    # Summary CSV
    pbar.set_description("  Saving › summary CSV")
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        os.path.join(out_dir, tasks[0]))
    pbar.update(1)

    # Per-class CSV
    pbar.set_description("  Saving › per-class CSV")
    per_class_df.to_csv(os.path.join(out_dir, tasks[1]), index=False)
    pbar.update(1)

    # JSON
    pbar.set_description("  Saving › metrics JSON")
    with open(os.path.join(out_dir, tasks[2]), "w") as f:
        json.dump({
            "summary": {
                k: (round(float(v), 6)
                    if isinstance(v, (float, np.floating)) else v)
                for k, v in summary.items()
            },
            "per_class": per_class_df.to_dict(orient="records"),
        }, f, indent=2)
    pbar.update(1)
    pbar.close()

    log.info(f"Outputs → {out_dir}/{tasks[0]}")
    log.info(f"Outputs → {out_dir}/{tasks[1]}")
    log.info(f"Outputs → {out_dir}/{tasks[2]}")

# ==============================================================================
# 9.  MAIN
# ==============================================================================
def evaluate(split: str):
    print(f"\n{'='*62}")
    print(f"  LandCoverAI Evaluator  |  split={split.upper()}  |  device={DEVICE}")
    print(f"{'='*62}\n")

    split_type = detect_split_type(DATASET_ROOT, split)
    log.info(f"Split type auto-detected: {split_type.upper()}")

    # ── Step 1 : Load model ────────────────────────────────────────────────────
    print("[1/5] Loading model…")
    model      = load_model(CHECKPOINT)
    complexity = model_complexity(model)
    log.info(f"Model complexity: {complexity}")

    if split_type == "supervised":
        # ── Step 2 : Build dataset ─────────────────────────────────────────────
        print("[2/5] Building dataset…")
        dataset = SupervisedDataset(DATASET_ROOT, split)
        loader  = torch.utils.data.DataLoader(
            dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,           # 0 = safe on Windows
            pin_memory  = False,
        )

        # ── Step 3 : Inference ─────────────────────────────────────────────────
        print(f"[3/5] Running inference on {len(dataset)} images…")
        cm_accum, probs_sample, gts_sample, perf = run_supervised(model, loader)
        perf.update(complexity)

        # ── Step 4 : Compute metrics ───────────────────────────────────────────
        print("[4/5] Computing metrics…")
        summary, per_class_df, cm = compute_all_metrics(
            cm_accum, probs_sample, gts_sample, perf)

        # ── Console summary ────────────────────────────────────────────────────
        print(f"\n{'─'*62}")
        print(f"  RESULTS SUMMARY — {split.upper()}")
        print(f"{'─'*62}")
        for k, v in summary.items():
            bar = ""
            if isinstance(v, float) and 0 <= v <= 1:
                filled = int(v * 30)
                bar    = f"  [{'█'*filled}{'░'*(30-filled)}]"
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<30}  {val_str}{bar}")

        print(f"\n  PER-CLASS METRICS (excl. unknown):")
        print(per_class_df[per_class_df["Class"] != "unknown"]
              .set_index("Class")[["IoU","Dice","Precision","Recall","F1"]]
              .to_string())

        # ── Step 5 : Save outputs + plots ─────────────────────────────────────
        print("\n[5/5] Saving outputs and generating plots…")
        save_outputs(summary, per_class_df, OUTPUT_DIR, split)
        _save_plots(summary, per_class_df, cm,
                    probs_sample, gts_sample, OUTPUT_DIR, split)

    else:
        # ── Inference-only mode (val / test) ───────────────────────────────────
        print("[2/5] Building inference dataset…")
        dataset = InferenceDataset(DATASET_ROOT, split)
        loader  = torch.utils.data.DataLoader(
            dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = False,
            collate_fn  = lambda batch: (
                torch.stack([b[0] for b in batch]),
                [b[1] for b in batch],
                [b[2] for b in batch],
            ),
        )

        print(f"[3/5] Running inference on {len(dataset)} images…")
        perf = run_inference_only(model, loader, OUTPUT_DIR, split)
        perf.update(complexity)

        print(f"\n{'─'*62}")
        print(f"  INFERENCE COMPLETE — {split.upper()}  (no ground-truth masks)")
        print(f"{'─'*62}")
        for k, v in perf.items():
            print(f"  {k:<30}  {v:.4f}" if isinstance(v, float) else
                  f"  {k:<30}  {v}")
        print(f"\n  Overlays → eval_output/predictions_{split}/")
        print("  [4/5] and [5/5] skipped — no ground truth available.")

    print(f"\n✔ Done. All outputs saved to: {OUTPUT_DIR}/\n")


# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LandCoverAI Segmentation Evaluator")
    parser.add_argument("--split",      default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--dataset",    default=DATASET_ROOT,
                        help="Path to dataset root folder")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Path to best_model.pth")
    parser.add_argument("--batch",      type=int, default=BATCH_SIZE,
                        help="Batch size for inference")
    args = parser.parse_args()

    DATASET_ROOT = args.dataset
    CHECKPOINT   = args.checkpoint
    BATCH_SIZE   = args.batch

    evaluate(args.split)