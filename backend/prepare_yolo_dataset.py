"""
Prepares a YOLO-format detection dataset from the existing ImageFolder classification data.

Two annotation modes
--------------------
Grounding DINO (default): runs a zero-shot open-vocabulary detector on each image to
    produce tight bounding boxes — greatly improves multi-item scene generalisation.
Pseudo-label (--pseudo flag): sets every bbox to the full image (cx=0.5 cy=0.5 w=1.0 h=1.0).
    Fast, but the detector never learns to localise items smaller than the frame.

Run once before YOLO training:
    python prepare_yolo_dataset.py           # recommended
    python prepare_yolo_dataset.py --pseudo  # fast fallback
"""

import argparse
import subprocess
from pathlib import Path

import torch
from PIL import Image

BASE       = Path(__file__).parent / "data"
YOLO_DIR   = BASE / "yolo"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GDINO_MODEL = "IDEA-Research/grounding-dino-base"

train_classes = sorted([d.name for d in (BASE / "train").iterdir() if d.is_dir()])
class_to_idx  = {c: i for i, c in enumerate(train_classes)}


def _make_junction(link: Path, target: Path) -> None:
    """Create a Windows directory junction; raise if it already exists but points elsewhere."""
    resolved = target.resolve()
    if link.exists() or link.is_symlink():
        actual = link.resolve()
        if actual != resolved:
            raise RuntimeError(
                f"Junction {link} already points to {actual!r}; expected {resolved!r}. "
                "Delete it manually and re-run."
            )
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(resolved)],
        check=True, capture_output=True,
    )


def _class_to_prompt(class_name: str) -> str:
    """Strip fresh/rotten prefix so GDINO gets a clean produce-type prompt, e.g. 'apple.'"""
    name = class_name.lower()
    for prefix in ("fresh_", "rotten_", "fresh", "rotten"):
        if name.startswith(prefix):
            name = name[len(prefix):].lstrip("_")
            break
    return name.replace("_", " ").strip() + "."


def _load_gdino(device: torch.device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print(f"Loading Grounding DINO ({GDINO_MODEL}) …")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
    model.eval()
    return processor, model


def _annotate_gdino(
    img_path: Path,
    class_id: int,
    class_name: str,
    processor,
    model,
    device: torch.device,
    threshold: float = 0.3,
) -> str:
    """Return YOLO-format label lines for one image using Grounding DINO."""
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    text = _class_to_prompt(class_name)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"]
    if len(boxes) == 0:
        return f"{class_id} 0.5 0.5 1.0 1.0\n"

    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cx = max(0.0, min(1.0, (x1 + x2) / 2 / w))
        cy = max(0.0, min(1.0, (y1 + y2) / 2 / h))
        bw = max(0.0, min(1.0, (x2 - x1) / w))
        bh = max(0.0, min(1.0, (y2 - y1) / h))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return "".join(lines)


def prepare(use_pseudo: bool = False) -> None:
    assert (BASE / "train").is_dir(), "data/train/ not found — run the split script first"
    assert (BASE / "val").is_dir(),   "data/val/ not found — run the split script first"

    _make_junction(YOLO_DIR / "images" / "train", BASE / "train")
    _make_junction(YOLO_DIR / "images" / "val",   BASE / "val")
    print("Image junctions ready.")

    gdino_processor = gdino_model = gdino_device = None
    if not use_pseudo:
        gdino_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gdino_processor, gdino_model = _load_gdino(gdino_device)

    total = fallback = 0
    for split in ("train", "val"):
        for class_dir in sorted((BASE / split).iterdir()):
            if not class_dir.is_dir():
                continue
            class_id  = class_to_idx[class_dir.name]
            label_dir = YOLO_DIR / "labels" / split / class_dir.name
            label_dir.mkdir(parents=True, exist_ok=True)

            for img in class_dir.iterdir():
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue

                if use_pseudo:
                    label_text = f"{class_id} 0.5 0.5 1.0 1.0\n"
                else:
                    label_text = _annotate_gdino(
                        img, class_id, class_dir.name,
                        gdino_processor, gdino_model, gdino_device,
                    )
                    fallback_line = f"{class_id} 0.5 0.5 1.0 1.0\n"
                    if label_text == fallback_line:
                        fallback += 1

                (label_dir / (img.stem + ".txt")).write_text(label_text)
                total += 1

        print(f"  [{split}] labels written.")

    print(f"Total label files: {total:,}")
    if not use_pseudo:
        print(f"  Fallback (no GDINO detection): {fallback:,} ({fallback / total:.1%})")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(train_classes))
    yaml_content = (
        f"path: {YOLO_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(train_classes)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"dataset.yaml written → {yaml_path}")
    print(f"Classes: {len(train_classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pseudo", action="store_true",
        help="Use full-image pseudo-labels instead of Grounding DINO annotations",
    )
    args = parser.parse_args()
    prepare(use_pseudo=args.pseudo)
