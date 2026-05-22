"""
Fruit & Vegetable Detector
Uses YOLOv8m (ultralytics) + OpenCV + Pillow to detect produce in images,
draw bounding boxes, crop detections to 640px, and save them to ./uploads/
"""

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "models/best.pt"
UPLOAD_DIR   = Path("uploads")
CROP_SIZE    = 640                   # each saved crop is resized to 640×640
CONF_THRESH  = 0.30                  # minimum confidence to keep a detection

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_image(source: str) -> np.ndarray:
    """Load an image from a local path or a URL and return a BGR numpy array."""
    source = source.strip()

    if source.startswith(("http://", "https://")):
        print(f"  Downloading image from URL …")
        tmp = Path("_tmp_download.jpg")
        urllib.request.urlretrieve(source, tmp)
        img = cv2.imread(str(tmp))
        tmp.unlink(missing_ok=True)
    else:
        if not Path(source).exists():
            sys.exit(f"[ERROR] File not found: {source}")
        img = cv2.imread(source)

    if img is None:
        sys.exit("[ERROR] Could not decode image. Check the path/URL.")
    return img


def is_produce(label: str) -> bool:
    return label.lower() in PRODUCE_CLASSES


def make_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def unique_path(folder: Path, stem: str, ext: str = ".jpg") -> Path:
    """Return a non-colliding filepath like uploads/apple_0.jpg, apple_1.jpg …"""
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

    text     = f"{label} {conf:.0%}"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.65
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
    """Crop detection, resize to CROP_SIZE × CROP_SIZE, save as JPEG."""
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert BGR → RGB for Pillow
    crop_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop  = Image.fromarray(crop_rgb)
    pil_crop  = pil_crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)

    save_path = unique_path(folder, label.lower().replace(" ", "_"))
    pil_crop.save(save_path, quality=92)
    return save_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   🍎  Fruit & Vegetable Detector  (YOLOv11m)")
    print("=" * 58)

    source = input("\nEnter image path or URL: ").strip()
    if not source:
        sys.exit("[ERROR] No input provided.")

    # 1. Load image
    print("\n[1/4] Loading image …")
    img_bgr = load_image(source)
    h, w    = img_bgr.shape[:2]
    print(f"      Resolution: {w} × {h} px")

    # 2. Load / download model
    print(f"\n[2/4] Loading model ({MODEL_NAME}) …")
    model = YOLO(MODEL_NAME)          # auto-downloads on first use

    # 3. Run inference
    print("\n[3/4] Running inference …")
    results = model(img_bgr, conf=CONF_THRESH, verbose=False)[0]

    annotated = img_bgr.copy()
    folder    = make_upload_dir()
    saved     = []
    skipped   = 0

    print("\n[4/4] Processing detections …\n")
    print(f"  {'Label':<20} {'Conf':>6}   Box (x1,y1,x2,y2)")
    print("  " + "-" * 55)

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if not is_produce(label):
            skipped += 1
            continue

        color = BOX_COLORS[i % len(BOX_COLORS)]
        draw_box(annotated, x1, y1, x2, y2, label, conf, color)

        save_path = crop_and_save(img_bgr, x1, y1, x2, y2, label, folder)
        if save_path:
            saved.append((label, conf, save_path))
            print(f"  {label:<20} {conf:>5.1%}   ({x1},{y1}) → ({x2},{y2})")
            print(f"  {'':20}          → saved: {save_path}")

    # 4. Show annotated image
    window_title = "Fruit & Vegetable Detection  [press any key to close]"
    cv2.imshow(window_title, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Also save the full annotated image
    ann_path = folder / "annotated_result.jpg"
    cv2.imwrite(str(ann_path), annotated)

    # Summary
    print("\n" + "=" * 58)
    print(f"  Detections kept  : {len(saved)}")
    print(f"  Non-produce skip : {skipped}")
    print(f"  Crops saved to   : {folder.resolve()}/")
    print(f"  Annotated image  : {ann_path.resolve()}")
    print("=" * 58)

    if not saved:
        print("\n  ⚠  No fruits or vegetables detected.")
        print("     Try a different image or lower CONF_THRESH in the script.")


if __name__ == "__main__":
    main()