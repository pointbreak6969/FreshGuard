"""
YOLO11m fine-tuning script — LVIS Fruits & Vegetables (63 classes)


Usage:
  python prepare_yolo.py                        # default settings
  python prepare_yolo.py --epochs 150 --batch 8 # lower batch if OOM
  python prepare_yolo.py --resume               # continue interrupted run
  python prepare_yolo.py --export onnx          # train + export to ONNX
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent
DATA_YAML = BASE_DIR / "data" / "lvis_fruits_and_vegetables" / "data.yaml"
WEIGHTS   = BASE_DIR / "yolo11m.pt"
RUNS_DIR  = BASE_DIR / "runs" / "train"
RUN_NAME  = "lvis_finetune"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11m on LVIS fruits & vegetables")
    p.add_argument("--epochs",  type=int,   default=100,
                   help="Total training epochs (default: 100)")
    p.add_argument("--batch",   type=int,   default=16,
                   help="Batch size; drop to 8 if you hit OOM (default: 16)")
    p.add_argument("--imgsz",   type=int,   default=640,
                   help="Training image size in pixels (default: 640)")
    p.add_argument("--freeze",  type=int,   default=10,
                   help="Freeze first N backbone layers; 0 = full fine-tune (default: 10)")
    p.add_argument("--resume",  action="store_true",
                   help="Resume the last interrupted run")
    p.add_argument("--export",  choices=["onnx", "tensorrt", "none"], default="none",
                   help="Export best weights after training (default: none)")
    return p.parse_args()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(DATA_YAML),

        # ── duration & stopping ──────────────────────────────────────────────
        epochs=args.epochs,
        patience=20,            # stop early if mAP doesn't improve for 20 epochs

        # ── hardware ─────────────────────────────────────────────────────────
        device=0,               # RTX 4060
        batch=args.batch,
        imgsz=args.imgsz,
        workers=4,              # 4 is safe on Windows; raise to 8 if CPU is idle
        cache=True,             # load dataset into RAM — fits easily in 16 GB DDR5

        # ── transfer learning ────────────────────────────────────────────────
        # Freeze the first 10 backbone layers so pretrained COCO features are
        # preserved. With ~1000 images across 63 classes (~16/class) the head
        # needs to adapt but the backbone is already good.
        freeze=args.freeze,

        # ── optimizer & LR schedule ──────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,              # initial LR
        lrf=0.01,               # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,            # cosine annealing

        # ── augmentation ─────────────────────────────────────────────────────
        # Ultralytics default augmentations (mosaic, flip, hsv, etc.) are left
        # on — they work well for produce detection out of the box.
        close_mosaic=10,        # disable mosaic in last 10 epochs for stability

        # ── logging & checkpoints ────────────────────────────────────────────
        project=str(RUNS_DIR),
        name=RUN_NAME,
        save=True,
        save_period=10,         # checkpoint every 10 epochs
        plots=True,             # save training curve plots
        verbose=True,

        resume=args.resume,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def validate(weights: Path):
    print(f"\nValidating {weights} ...")
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA_YAML), device=0, verbose=True)
    print(f"  mAP50   : {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


# ── Export ────────────────────────────────────────────────────────────────────

def export(weights: Path, fmt: str):
    print(f"\nExporting {weights} → {fmt} ...")
    model = YOLO(str(weights))
    # TensorRT export leverages your RTX 4060 for fastest inference
    model.export(format=fmt, device=0, imgsz=640)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLO11m  .  LVIS Fruits & Vegetables  .  63 classes")
    print("=" * 60)
    print(f"  Weights : {WEIGHTS}")
    print(f"  Data    : {DATA_YAML}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print(f"  Freeze  : first {args.freeze} layers  |  Device: RTX 4060 (cuda:0)")
    print("=" * 60)

    if not WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS}")
        print("        Place yolo11m.pt in the backend/ folder and retry.")
        return

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        return

    train(args)

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        # ultralytics appends a numeric suffix when the run name already exists
        candidates = sorted(RUNS_DIR.glob(f"{RUN_NAME}*/weights/best.pt"))
        if candidates:
            best = candidates[-1]

    if best.exists():
        validate(best)
        if args.export != "none":
            export(best, args.export)
        print(f"\nDone. Best weights saved to:\n  {best.resolve()}")
    else:
        print("\n[WARN] best.pt not found — training may have been interrupted.")


if __name__ == "__main__":
    main()
