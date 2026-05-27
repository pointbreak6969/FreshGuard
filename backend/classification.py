"""
FreshGuard Classifier
Loads the finetuned ResNet50 from models/ and classifies cropped detections
produced by detection.py (saved in uploads/).

Can also be imported and used directly:
    from classification import load_classifier, classify_image, get_transform
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_DIR   = Path("models")
WEIGHTS_PATH = MODELS_DIR / "freshguard_resnet50.pth"
CLASSES_PATH = MODELS_DIR / "class_names.json"
UPLOADS_DIR  = Path("uploads")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SIZE   = 232          # matches ResNet50 training config


# ── Setup ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier(device: torch.device):
    """Load ResNet50 weights and class names. Returns (model, class_names)."""
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_image(
    pil_img: Image.Image,
    model: torch.nn.Module,
    class_names: list,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[str, float]:
    """Classify a single PIL image. Returns (predicted_class, confidence)."""
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
    return class_names[idx.item()], conf.item()




def get_classified_result(crop_paths: list[str | Path]) -> list[dict]:
    """Classify a list of crop file paths.

    Args:
        crop_paths: list of file paths (str or Path) to cropped images.

    Returns:
        list of dicts with keys: file, label, confidence.
    """
    if not crop_paths:
        return []

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class names not found at {CLASSES_PATH}")

    device = get_device()
    model, class_names = load_classifier(device)
    transform = get_transform()

    results = []
    for crop_path in crop_paths:
        img = Image.open(crop_path).convert("RGB")
        label, conf = classify_image(img, model, class_names, transform, device)
        results.append({
            "file":       str(crop_path),
            "label":      label,
            "confidence": conf,
        })

    return results


