"""
Land Cover Segmentation — Predict & Generate Bounding Boxes
============================================================
Takes trained model, runs inference on valid/test images,
converts predicted segmentation masks into per-class bounding boxes
with confidence scores, and saves annotated output images.

Workflow:
  1. Load trained model checkpoint
  2. Run inference on each image → softmax probabilities + class mask
  3. Per class: find connected components in the mask
  4. For each component: compute bounding box + mean confidence (softmax prob)
  5. Filter out boxes below MIN_CONFIDENCE or MIN_AREA thresholds
  6. Draw labeled bounding boxes on the original image
  7. Save annotated images + optional JSON/CSV detections

Usage on Google Colab:
  Adjust DATA_DIR, CHECKPOINT_PATH, and OUTPUT_DIR, then run all cells.
"""

# ============================================================
# Cell 1: Imports & Configuration
# ============================================================
import os
import glob
import json
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────
DATA_DIR        = r"c:\Users\kocha\pr"
CHECKPOINT_PATH = r"c:\Users\kocha\pr\best_model.pth"
OUTPUT_DIR      = r"c:\Users\kocha\pr\bbox_results"

# Where the satellite images live (no masks needed)
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# ── Model Config ───────────────────────────────────────────
IMAGE_SIZE   = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Detection Thresholds ──────────────────────────────────
# Note:
#   - Increase MIN_CONFIDENCE and MIN_AREA to reduce clutter from many tiny boxes.
#   - NMS_IOU_THRESH controls how aggressively overlapping boxes are merged.
MIN_CONFIDENCE = 0.65   # minimum mean softmax probability to keep a bbox
MIN_AREA       = 2000   # minimum pixel area of a connected component
NMS_IOU_THRESH = 0.40   # IoU threshold for non-maximum suppression (per class)

# ── Class Definitions (from DeepGlobe class_dict.csv) ──────
# These MUST match what the model was trained with
CLASS_NAMES = [
    "urban_land",    # 0
    "agriculture",   # 1
    "rangeland",     # 2
    "forest",        # 3
    "water",         # 4
    "barren",        # 5
    "unknown",       # 6
]
NUM_CLASSES = len(CLASS_NAMES)

# Colors for bounding box drawing (BGR for OpenCV)
CLASS_COLORS_BGR = [
    (0, 255, 255),    # urban_land  → yellow
    (0, 255, 0),      # agriculture → green
    (0, 165, 255),    # rangeland   → orange
    (0, 100, 0),      # forest      → dark green
    (255, 0, 0),      # water       → blue
    (255, 255, 255),  # barren      → white
    (128, 128, 128),  # unknown     → gray
]

# Classes to skip detection for (e.g., 'unknown')
SKIP_CLASSES = {6}  # skip unknown class

print(f"Device: {DEVICE}")
print(f"Min confidence: {MIN_CONFIDENCE}")
print(f"Min area: {MIN_AREA} px")
print(f"Classes: {CLASS_NAMES}")


# ============================================================
# Cell 2: Load Model
# ============================================================
def load_model(checkpoint_path):
    """Load UnetPlusPlus + EfficientNet-B7 from checkpoint."""
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights=None,  # will load from checkpoint
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_attention_type="scse",
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Handle both full checkpoint dict and raw state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        miou  = checkpoint.get("miou", "?")
        print(f"✅ Loaded checkpoint from epoch {epoch}, mIoU={miou}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded raw state dict")

    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model(CHECKPOINT_PATH)


# ============================================================
# Cell 3: Inference Helpers
# ============================================================
# Preprocessing — must match training val_transform exactly
preprocess = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def predict_mask(model, image_rgb, device=DEVICE):
    """
    Run inference on a single RGB image.

    Returns:
        class_mask : np.ndarray (H, W) — predicted class index per pixel
        confidence : np.ndarray (H, W) — softmax probability of predicted class
        probs      : np.ndarray (C, H, W) — full softmax probabilities
    """
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize to model input size
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Preprocess
    aug = preprocess(image=resized)
    tensor = aug["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    # Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(tensor)  # (1, C, H, W)

    # Softmax probabilities
    probs = F.softmax(logits.float(), dim=1)  # (1, C, H, W)

    # Resize probs back to original image size
    probs = F.interpolate(probs, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
    probs = probs.squeeze(0).cpu().numpy()  # (C, H, W)

    # Class predictions
    class_mask = probs.argmax(axis=0)      # (H, W)
    confidence = probs.max(axis=0)         # (H, W) — confidence of winning class

    return class_mask, confidence, probs


# ============================================================
# Cell 4: Mask → Bounding Boxes Conversion
# ============================================================
def mask_to_bboxes(class_mask, confidence_map, probs,
                   min_confidence=MIN_CONFIDENCE,
                   min_area=MIN_AREA,
                   skip_classes=SKIP_CLASSES):
    """
    Convert a segmentation mask into a list of bounding boxes.

    For each class:
      1. Create binary mask of pixels predicted as that class
      2. Find connected components
      3. For each component, compute bounding box and mean confidence
      4. Filter by min_confidence and min_area

    Returns:
        list of dicts:
          {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": [x_min, y_min, x_max, y_max],
            "area": int,
          }
    """
    detections = []

    for class_id in range(NUM_CLASSES):
        if class_id in skip_classes:
            continue

        # Binary mask for this class
        binary = (class_mask == class_id).astype(np.uint8)

        # Light morphology to clean up tiny specks before connected components.
        # This helps avoid many minute boxes from isolated noisy pixels.
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if binary.sum() == 0:
            continue

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # stats columns: [x, y, width, height, area]
        for label_id in range(1, num_labels):  # skip background (label 0)
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            # Filter tiny regions
            if area < min_area:
                continue

            # Mean confidence of this class within this component
            component_mask = (labels == label_id)
            mean_conf = float(probs[class_id][component_mask].mean())

            # Filter low confidence
            if mean_conf < min_confidence:
                continue

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": round(mean_conf, 4),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
            })

    return detections


def nms_per_class(detections, iou_threshold=NMS_IOU_THRESH):
    """
    Apply non-maximum suppression per class to remove overlapping boxes.
    """
    if not detections:
        return []

    filtered = []
    # Group by class
    class_ids = set(d["class_id"] for d in detections)

    for cid in class_ids:
        class_dets = [d for d in detections if d["class_id"] == cid]

        if not class_dets:
            continue

        # Sort by confidence (descending)
        class_dets.sort(key=lambda d: d["confidence"], reverse=True)

        boxes = np.array([d["bbox"] for d in class_dets], dtype=np.float32)
        scores = np.array([d["confidence"] for d in class_dets], dtype=np.float32)

        # Compute IoU matrix
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        indices = list(range(len(class_dets)))

        while indices:
            i = indices[0]
            keep.append(i)
            indices = indices[1:]

            if not indices:
                break

            # Compute IoU with remaining
            remaining = np.array(indices)
            xx1 = np.maximum(x1[i], x1[remaining])
            yy1 = np.maximum(y1[i], y1[remaining])
            xx2 = np.minimum(x2[i], x2[remaining])
            yy2 = np.minimum(y2[i], y2[remaining])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter_area = inter_w * inter_h

            iou = inter_area / (areas[i] + areas[remaining] - inter_area + 1e-6)

            # Keep only low-overlap boxes
            indices = [indices[j] for j in range(len(indices)) if iou[j] < iou_threshold]

        filtered.extend([class_dets[k] for k in keep])

    return filtered


# ============================================================
# Cell 5: Visualization — Draw Bounding Boxes
# ============================================================
def draw_bboxes(image_bgr, detections, thickness=3, font_scale=0.7):
    """
    Draw labeled bounding boxes with confidence on the image.

    Args:
        image_bgr : np.ndarray (H, W, 3) — BGR image
        detections: list of detection dicts from mask_to_bboxes
        thickness : bbox line thickness
        font_scale: label font size

    Returns:
        annotated : np.ndarray — image with drawn bounding boxes
    """
    annotated = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        conf     = det["confidence"]
        name     = det["class_name"]
        color    = CLASS_COLORS_BGR[class_id]

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label with background
        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        # Ensure label stays within image bounds
        label_y1 = max(y1 - th - baseline - 6, 0)
        label_y2 = max(y1, th + baseline + 6)

        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)

        # Text color — use black for bright backgrounds, white for dark
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        cv2.putText(
            annotated, label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, 2, cv2.LINE_AA
        )

    return annotated


def draw_mask_overlay(image_bgr, class_mask, alpha=0.35):
    """
    Draw semi-transparent segmentation mask overlay on the image.
    """
    overlay = image_bgr.copy()
    for class_id in range(NUM_CLASSES):
        if class_id in SKIP_CLASSES:
            continue
        mask = (class_mask == class_id)
        if mask.any():
            color_rgb = CLASS_COLORS_BGR[class_id]
            overlay[mask] = color_rgb

    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    return blended


# ============================================================
# Cell 6: Process a Single Image (end-to-end)
# ============================================================
def process_image(model, image_path, output_dir, save_json=True):
    """
    Full pipeline for one image:
      1. Read image
      2. Predict mask
      3. Extract bounding boxes
      4. Apply NMS
      5. Draw and save annotated image
      6. Optionally save detection JSON

    Returns:
        detections: list of detection dicts
    """
    # Read image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ⚠️ Cannot read: {image_path}")
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Predict
    class_mask, confidence_map, probs = predict_mask(model, image_rgb)

    # Extract bounding boxes
    detections = mask_to_bboxes(class_mask, confidence_map, probs)

    # Apply NMS
    detections = nms_per_class(detections)

    # Sort by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ── Save annotated images ─────────────────────────────
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Bounding box image
    bbox_img = draw_bboxes(image_bgr, detections)
    bbox_path = os.path.join(output_dir, f"{basename}_bbox.jpg")
    cv2.imwrite(bbox_path, bbox_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 2. Mask overlay image
    overlay_img = draw_mask_overlay(image_bgr, class_mask)
    overlay_path = os.path.join(output_dir, f"{basename}_mask.jpg")
    cv2.imwrite(overlay_path, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 3. Combined: mask overlay + bounding boxes
    combined = draw_bboxes(overlay_img, detections)
    combined_path = os.path.join(output_dir, f"{basename}_combined.jpg")
    cv2.imwrite(combined_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 4. Save detections as JSON
    if save_json:
        json_path = os.path.join(output_dir, f"{basename}_detections.json")
        with open(json_path, "w") as f:
            json.dump({
                "image": os.path.basename(image_path),
                "image_size": [image_bgr.shape[1], image_bgr.shape[0]],
                "num_detections": len(detections),
                "detections": detections,
            }, f, indent=2)

    return detections


# ============================================================
# Cell 7: Run on Validation + Test Sets
# ============================================================
def run_prediction(model, image_dir, split_name, output_base=OUTPUT_DIR):
    """
    Run prediction + bbox extraction on all images in a directory.

    Args:
        model      : loaded PyTorch model
        image_dir  : directory containing *_sat.jpg images
        split_name : "valid" or "test" (for organizing output)
        output_base: base output directory

    Returns:
        all_detections: dict mapping image_name → list of detections
    """
    output_dir = os.path.join(output_base, split_name)
    os.makedirs(output_dir, exist_ok=True)

    # Find all satellite images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*_sat.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {len(image_paths)} images")
    print(f"Output → {output_dir}")
    print(f"{'='*60}")

    all_detections = {}
    total_boxes = 0
    class_counts = {name: 0 for name in CLASS_NAMES}

    for img_path in tqdm(image_paths, desc=f"Predict [{split_name}]"):
        detections = process_image(model, img_path, output_dir)
        img_name = os.path.basename(img_path)
        all_detections[img_name] = detections
        total_boxes += len(detections)

        for det in detections:
            class_counts[det["class_name"]] += 1

    # ── Summary Statistics ─────────────────────────────────
    print(f"\n📊 {split_name.upper()} SUMMARY")
    print(f"   Total images processed: {len(image_paths)}")
    print(f"   Total bounding boxes:   {total_boxes}")
    print(f"   Avg boxes per image:    {total_boxes / max(1, len(image_paths)):.1f}")
    print(f"\n   Per-class detection counts:")
    for name, count in class_counts.items():
        if name not in [CLASS_NAMES[i] for i in SKIP_CLASSES]:
            print(f"     {name:>15s}: {count}")

    # ── Save summary CSV ───────────────────────────────────
    csv_path = os.path.join(output_dir, f"{split_name}_all_detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "confidence",
                         "x_min", "y_min", "x_max", "y_max", "area"])
        for img_name, dets in all_detections.items():
            for det in dets:
                writer.writerow([
                    img_name,
                    det["class_id"],
                    det["class_name"],
                    det["confidence"],
                    *det["bbox"],
                    det["area"],
                ])

    print(f"\n   📁 CSV saved: {csv_path}")
    print(f"   📁 Annotated images saved to: {output_dir}/")

    return all_detections


# ============================================================
# Cell 8: Run Everything
# ============================================================
print(f"\n🚀 Starting prediction pipeline...")
print(f"   Checkpoint:      {CHECKPOINT_PATH}")
print(f"   Min confidence:  {MIN_CONFIDENCE}")
print(f"   Min area:        {MIN_AREA} px")
print(f"   NMS IoU thresh:  {NMS_IOU_THRESH}")

# Run on validation set
valid_detections = run_prediction(model, VALID_DIR, "valid")

# Run on test set
test_detections = run_prediction(model, TEST_DIR, "test")

# ============================================================
# Cell 9: Print Example Detections
# ============================================================
print(f"\n{'='*60}")
print("📋 EXAMPLE DETECTIONS (first 5 images from valid)")
print(f"{'='*60}")

for img_name, dets in list(valid_detections.items())[:5]:
    print(f"\n🖼️  {img_name}  ({len(dets)} detections)")
    for det in dets[:10]:  # show at most 10 per image
        x1, y1, x2, y2 = det["bbox"]
        print(f"   [{det['class_name']:>12s}]  conf={det['confidence']:.3f}  "
              f"bbox=({x1},{y1},{x2},{y2})  area={det['area']}")
    if len(dets) > 10:
        print(f"   ... and {len(dets) - 10} more")


# ============================================================
# Cell 10 (Optional): Visualize a Few Results in Colab
# ============================================================
# Uncomment to display results inline in a Colab notebook:
#
# import matplotlib.pyplot as plt
#
# def show_results(image_dir, output_dir, n=4):
#     """Display original + bbox + mask overlay for n sample images."""
#     image_paths = sorted(glob.glob(os.path.join(image_dir, "*_sat.jpg")))[:n]
#
#     fig, axes = plt.subplots(n, 3, figsize=(20, 6 * n))
#     if n == 1:
#         axes = axes[np.newaxis, :]
#
#     for i, img_path in enumerate(image_paths):
#         basename = os.path.splitext(os.path.basename(img_path))[0]
#
#         original  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         bbox_img  = cv2.cvtColor(
#             cv2.imread(os.path.join(output_dir, f"{basename}_bbox.jpg")),
#             cv2.COLOR_BGR2RGB
#         )
#         combined  = cv2.cvtColor(
#             cv2.imread(os.path.join(output_dir, f"{basename}_combined.jpg")),
#             cv2.COLOR_BGR2RGB
#         )
#
#         axes[i, 0].imshow(original)
#         axes[i, 0].set_title("Original", fontsize=14)
#         axes[i, 0].axis("off")
#
#         axes[i, 1].imshow(bbox_img)
#         axes[i, 1].set_title("Bounding Boxes", fontsize=14)
#         axes[i, 1].axis("off")
#
#         axes[i, 2].imshow(combined)
#         axes[i, 2].set_title("Mask + BBoxes", fontsize=14)
#         axes[i, 2].axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
# show_results(VALID_DIR, os.path.join(OUTPUT_DIR, "valid"), n=4)
# show_results(TEST_DIR,  os.path.join(OUTPUT_DIR, "test"),  n=4)

print("\n✅ Pipeline complete!")
