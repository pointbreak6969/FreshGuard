"""
Fruit & Vegetable Detector
Uses YOLOv8m (ultralytics) + OpenCV + Pillow to detect produce in images,
draw bounding boxes, crop detections to 640px, and save them to ./uploads/
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# -- Configuration ------------------------------------------------------------

MODEL_NAME  = "models/best.pt"
UPLOAD_DIR  = Path("uploads")
CROP_SIZE   = 480    # canvas size; crop is fitted inside while preserving aspect ratio
CONF_THRESH = 0.30   # minimum confidence to keep a detection

# All 63 classes from the fine-tuned LVIS model (matches data.yaml names exactly)
PRODUCE_CLASSES = {
    "almond", "apple", "apricot", "artichoke", "asparagus", "avocado",
    "banana", "tofu", "bell pepper", "blackberry", "blueberry", "broccoli",
    
    "brussels sprouts", "cantaloupe", "carrot", "cauliflower", "cayenne pepper",
    "celery", "cherry", "chickpea", "chili pepper", "clementine", "coconut",
    "corn", "cucumber", "date", "eggplant", "fig", "garlic", "ginger",
    "gourd", "grape", "green bean", "green onion", "kiwi", "lemon", "lettuce",
    "lime", "mandarin orange", "melon", "mushroom", "onion", "orange", "papaya",
    "pea", "peach", "pear", "persimmon", "pickle", "pineapple", "potato",
    "prune", "pumpkin", "radish", "raspberry", "strawberry", "sweet potato",
    "tomato", "turnip", "watermelon", "zucchini",
}

# Vivid color palette for bounding boxes (BGR)
BOX_COLORS = [
    (0, 220, 90),   (0, 150, 255),  (255, 80, 0),
    (200, 0, 255),  (0, 230, 230),  (255, 200, 0),
]


# -- Helpers ------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_produce(label: str) -> bool:
    return label.lower() in PRODUCE_CLASSES


def make_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def unique_path(folder: Path, stem: str, ext: str = ".png") -> Path:
    """Return a non-colliding filepath like uploads/apple_0.png, apple_1.png ..."""
    idx = 0
    while True:
        p = folder / f"{stem}_{idx}{ext}"
        if not p.exists():
            return p
        idx += 1


def draw_box(img: np.ndarray, x1, y1, x2, y2,
             label: str, conf: float, color: tuple) -> None:
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    text      = f"{label} {conf:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.65
    txt_thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, txt_thick)

    # Filled background for the label
    by1 = max(y1 - th - baseline - 6, 0)
    by2 = y1
    cv2.rectangle(img, (x1, by1), (x1 + tw + 6, by2), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - baseline - 2),
                font, scale, (255, 255, 255), txt_thick, cv2.LINE_AA)


def crop_and_save(img_bgr: np.ndarray, x1, y1, x2, y2,
                  label: str, folder: Path) -> Path:
    """Crop detection, resize to fit within CROP_SIZE while preserving aspect ratio, save as PNG."""
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert BGR -> RGB for Pillow
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)

    # Resize to fit within CROP_SIZE while preserving aspect ratio
    pil_crop.thumbnail((CROP_SIZE, CROP_SIZE), Image.LANCZOS)

    save_path = unique_path(folder, label.lower().replace(" ", "_"))
    pil_crop.save(save_path)  # PNG — lossless, no quality param needed
    return save_path


def annotated_output_path(folder: Path, output_base: str | Path | None) -> Path:
    if not output_base:
        return folder / "annotated_result.jpg"

    base = Path(output_base)
    if base.suffix:
        base = base.with_suffix("")
    return base.parent / f"{base.name}_annotated.jpg"


# -- API ----------------------------------------------------------------------

def detect(
    pil_img: Image.Image,
    output_base: str | Path | None = None,
) -> tuple[list[dict], Path]:
    """Run detection on a PIL image and return detections and annotated path.

    Each detection is a dict with: label, confidence, path.
    """
    pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    device = get_device()
    model = YOLO(MODEL_NAME)
    results = model(img_bgr, conf=CONF_THRESH, verbose=False, device=device)[0]

    annotated = img_bgr.copy()
    folder = make_upload_dir()
    detections = []

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if not is_produce(label):
            continue

        color = BOX_COLORS[i % len(BOX_COLORS)]
        draw_box(annotated, x1, y1, x2, y2, label, conf, color)

        save_path = crop_and_save(img_bgr, x1, y1, x2, y2, label, folder)
        if save_path:
            detections.append({
                "label":      label,
                "confidence": conf,
                "path":       str(save_path),
            })

    ann_path = annotated_output_path(folder, output_base)
    cv2.imwrite(str(ann_path), annotated)

    return detections, ann_path
