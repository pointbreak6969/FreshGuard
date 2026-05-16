import os
import torch
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────
# Finetuned YOLO detects bounding boxes AND classifies Fresh/Rotten + species
# in a single forward pass — no separate classifier stage needed.

_WEIGHTS = Path(__file__).parent / "models" / "freshguard_yolo" / "weights" / "best.pt"

_detector = YOLO(str(_WEIGHTS))

# ── Public API ────────────────────────────────────────────────────────────────

def detect(image: Image.Image, output_dir: str, threshold: float = 0.4) -> list[dict]:
    """
    Detect and classify all produce items in a multi-item scene.

    Returns a list of dicts with keys:
      label       — Fresh/Rotten class name (e.g. "FreshApple", "rottenbanana")
      confidence  — YOLO box confidence score
      path        — saved crop image path
    """
    results = _detector(image, imgsz=640, conf=threshold, verbose=False)[0]

    os.makedirs(output_dir, exist_ok=True)
    label_idx: dict[str, int] = {}
    detections: list[dict] = []

    for box in results.boxes:
        class_name = results.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        crop = image.crop([x1, y1, x2, y2])
        ratio = 720 / crop.height
        crop_display = crop.resize((int(crop.width * ratio), 720), Image.LANCZOS)

        idx = label_idx.get(class_name, 0)
        label_idx[class_name] = idx + 1
        crop_path = os.path.join(output_dir, f"{class_name}_{idx}.jpg")
        crop_display.save(crop_path)

        detections.append({
            "label": class_name,
            "confidence": round(float(box.conf), 2),
            "path": crop_path,
        })

    return detections
