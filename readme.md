# FreshGuard — Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset & Class Coverage](#3-dataset--class-coverage)
4. [Stage 1 — Object Detection (YOLO11m Fine-tuning)](#4-stage-1--object-detection-yolo11m-fine-tuning)
5. [Stage 2 — Freshness Classification (ResNet50 Fine-tuning)](#5-stage-2--freshness-classification-resnet50-fine-tuning)
6. [CNN Model Comparison](#6-cnn-model-comparison)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Frontend](#8-frontend)
9. [Limitations](#9-limitations)

---

## 1. Project Overview

FreshGuard is a two-stage produce freshness detection system. A user uploads a photo of fruits or vegetables through a web interface; the system locates each item in the image, classifies it as **fresh** or **rotten**, and returns a confidence score per item — all in a single API call.

**Tech stack**

| Layer | Technology |
|---|---|
| Detection | YOLO11m (Ultralytics), fine-tuned on LVIS Fruits & Vegetables |
| Classification | ResNet50, fine-tuned on a 34-class fresh/rotten dataset |
| Backend API | FastAPI (Python) |
| Frontend | Next.js 16 + React 19, Tailwind CSS, shadcn/ui |
| Hardware (training) | NVIDIA GeForce RTX 4060 (8 GB VRAM) |

---

## 2. System Architecture

```
User uploads image
        │
        ▼
  POST /detect  (FastAPI)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1 — Detection        │
│  YOLO11m  (best.pt)         │
│  63 produce classes         │
│  Confidence threshold: 0.30 │
│  Crops each detection       │
│  to 480 px canvas           │
└─────────┬───────────────────┘
          │  crop PNGs
          ▼
┌─────────────────────────────┐
│  Stage 2 — Classification   │
│  ResNet50  (freshguard_     │
│  resnet50.pth)              │
│  34 Fresh/Rotten classes    │
│  Input size: 232 × 232      │
│  Softmax confidence         │
└─────────┬───────────────────┘
          │
          ▼
  JSON response
  { image_id, results: [{file, label, confidence}] }
        │
        ▼
  Frontend renders
  colour-coded freshness badges
```

---

## 3. Dataset & Class Coverage

### 3.1 Why fine-tuning was necessary

Both models ship pre-trained on broad, general datasets:

- **YOLO11m** is pre-trained on **COCO** (80 classes). Of those 80, only a handful overlap with produce — `banana`, `apple`, `orange`, `broccoli`, `carrot`. The remaining 58+ COCO classes are vehicles, furniture, electronics, and animals. The vast majority of the 63 produce-specific classes needed for FreshGuard simply **did not exist** in the original COCO vocabulary.

- **ResNet50** is pre-trained on **ImageNet-1K** (~1,000 coarse categories). While it recognises generic fruit shapes, it has no concept of *freshness*. There is no "RottenMango" or "FreshGuava" class in ImageNet. The entire 34-class fresh/rotten label space was **absent** and had to be learned from scratch via fine-tuning.

### 3.2 YOLO Detection Classes (63 total — LVIS Fruits & Vegetables)

The fine-tuned YOLO model detects the following 63 categories. Items marked `†` are among the few that already appeared in COCO; everything else was a **new class** added through fine-tuning.

| ID | Class | ID | Class | ID | Class |
|---|---|---|---|---|---|
| 0 | almond | 22 | eggplant | 44 | pear |
| 1 | apple `†` | 23 | fig | 45 | persimmon |
| 2 | apricot | 24 | garlic | 46 | pickle |
| 3 | artichoke | 25 | ginger | 47 | pineapple |
| 4 | asparagus | 26 | gourd | 48 | potato |
| 5 | avocado | 27 | grape | 49 | prune |
| 6 | banana `†` | 28 | green bean | 50 | pumpkin |
| 7 | tofu | 29 | green onion | 51 | radish |
| 8 | bell pepper | 30 | kiwi | 52 | raspberry |
| 9 | blackberry | 31 | lemon | 53 | strawberry |
| 10 | blueberry | 32 | lettuce | 54 | sweet potato |
| 11 | broccoli `†` | 33 | lime | 55 | tomato `†` |
| 12 | brussels sprouts | 34 | mandarin orange | 56 | turnip |
| 13 | cantaloupe | 35 | tomato (dup.) | 57 | strawberry (dup.) |
| 14 | carrot `†` | 36 | melon | 58 | watermelon |
| 15 | cauliflower | 37 | mushroom | 59 | tomato (dup.) |
| 16 | cayenne pepper | 38 | onion | 60 | zucchini |
| 17 | celery | 39 | orange `†` | 61–62 | (padding) |
| 18 | cherry | 40 | papaya | | |
| 19 | chickpea | 41 | pea | | |
| 20 | chili pepper | 42 | peach | | |
| 21 | clementine | 43 | date | | |
| 22 | coconut | 44 | corn | | |

> **Note:** The LVIS source annotations contain duplicate IDs for `strawberry` (IDs 30 & 57) and `tomato` (IDs 35, 55 & 59) due to labelling inconsistencies in the upstream dataset.

### 3.3 Classification Classes (34 total — Fresh / Rotten)

The ResNet50 classifier recognises only the 17 specific produce items below. Any vegetable or fruit **detected by YOLO but not present in this list** will be given the nearest-match label by the classifier — a key source of error discussed in [Limitations](#9-limitations).

| # | Fresh Class | Rotten Class |
|---|---|---|
| 1 | FreshApple | rottenapples |
| 2 | FreshBellpepper | RottenBellpepper |
| 3 | FreshCarrot | RottenCarrot |
| 4 | FreshGrape | RottenGrape |
| 5 | FreshGuava | RottenGuava |
| 6 | FreshJujube | RottenJujube |
| 7 | FreshMango | RottenMango |
| 8 | FreshOrange | RottenOrange (rottenoranges) |
| 9 | FreshPomegranate | RottenPomegranate |
| 10 | FreshStrawberry | RottenStrawberry |
| 11 | freshbanana | rottenbanana |
| 12 | freshbittergroud | rottenbittergroud |
| 13 | freshcapsicum | rottencapsicum |
| 14 | freshcucumber | rottencucumber |
| 15 | freshokra | rottenokra |
| 16 | freshpotato | rottenpotato |
| 17 | freshtomato | rottentomato |

---

## 4. Stage 1 — Object Detection (YOLO11m Fine-tuning)

### 4.1 Training Configuration

| Parameter | Value |
|---|---|
| Base model | `yolo11m.pt` (COCO pre-trained) |
| Dataset | LVIS Fruits & Vegetables (63 classes) |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate (initial) | 0.001 |
| LR scheduler | Cosine annealing (lrf=0.01) |
| Warmup epochs | 3 |
| Frozen backbone layers | 10 (first 10 layers frozen) |
| Max epochs | 100 |
| Early stopping patience | 20 epochs |
| Augmentation | Mosaic, horizontal flip, HSV jitter, random erasing (p=0.4) |
| AMP | Enabled |
| Hardware | RTX 4060, CUDA |

### 4.2 Before vs. After Fine-tuning

The base YOLO11m model recognises only 5 produce-adjacent COCO classes. Without fine-tuning it achieves effectively **zero mAP** on the 63-class LVIS produce benchmark because 58+ classes are unknown vocabulary.

After fine-tuning on the LVIS Fruits & Vegetables dataset:

| Metric | Epoch 1 (start) | Best (Epoch 25) | Final (Epoch 45) |
|---|---|---|---|
| mAP@50 | 0.081 | **0.275** | 0.217 |
| mAP@50–95 | 0.045 | **0.178** | 0.126 |
| Precision | 0.277 | 0.469 | 0.401 |
| Recall | 0.204 | 0.269 | 0.249 |
| Train Box Loss | 1.387 | 1.169 | 1.082 |
| Train Cls Loss | 2.227 | 1.119 | 0.878 |
| Train DFL Loss | 1.184 | 1.083 | 1.037 |

Training ran for **45 epochs** before early-stopping triggered (best weights found at epoch 25, no improvement for 20 subsequent epochs). The saved `best.pt` corresponds to the epoch 25 checkpoint.

> **Reading the table:** mAP50 improved **3.4×** from epoch 1 to peak (0.081 → 0.275). Classification loss dropped **60%** (2.23 → 0.878), reflecting the model learning the new produce vocabulary. The slight regression in the final epoch vs. best is normal — the best weights are saved separately.

### 4.3 Training Curves

**Overall training progress (losses + mAP over all epochs):**

![Training Results](backend/runs/train/lvis_finetune/results.png)

**Precision curve across confidence thresholds:**

![Precision Curve](backend/runs/train/lvis_finetune/BoxP_curve.png)

**Recall curve across confidence thresholds:**

![Recall Curve](backend/runs/train/lvis_finetune/BoxR_curve.png)

**Precision–Recall curve:**

![PR Curve](backend/runs/train/lvis_finetune/BoxPR_curve.png)

**F1 score curve:**

![F1 Curve](backend/runs/train/lvis_finetune/BoxF1_curve.png)

### 4.4 Confusion Matrix

The normalised confusion matrix below shows per-class detection accuracy. Visually darker diagonal entries indicate stronger per-class recall, while off-diagonal entries reveal which classes are most commonly confused.

![Confusion Matrix Normalised](backend/runs/train/lvis_finetune/confusion_matrix_normalized.png)

![Confusion Matrix (counts)](backend/runs/train/lvis_finetune/confusion_matrix.png)

### 4.5 Training & Validation Sample Predictions

**Training batches (ground truth labels):**

![Train Batch 0](backend/runs/train/lvis_finetune/train_batch0.jpg)
![Train Batch 1](backend/runs/train/lvis_finetune/train_batch1.jpg)
![Train Batch 2](backend/runs/train/lvis_finetune/train_batch2.jpg)

**Validation: ground truth vs. model predictions:**

| Ground Truth | Predictions |
|---|---|
| ![Val Batch 0 Labels](backend/runs/train/lvis_finetune/val_batch0_labels.jpg) | ![Val Batch 0 Preds](backend/runs/train/lvis_finetune/val_batch0_pred.jpg) |
| ![Val Batch 1 Labels](backend/runs/train/lvis_finetune/val_batch1_labels.jpg) | ![Val Batch 1 Preds](backend/runs/train/lvis_finetune/val_batch1_pred.jpg) |
| ![Val Batch 2 Labels](backend/runs/train/lvis_finetune/val_batch2_labels.jpg) | ![Val Batch 2 Preds](backend/runs/train/lvis_finetune/val_batch2_pred.jpg) |

**Class label distribution in training data:**

![Class Distribution](backend/runs/train/lvis_finetune/labels.jpg)

### 4.6 Post-training Validation Run

An independent `yolo val` run was executed against the test split after training completed. The curves below are from that standalone evaluation and confirm the best-checkpoint generalisation:

![Val F1 Curve](backend/runs/detect/val/BoxF1_curve.png)
![Val PR Curve](backend/runs/detect/val/BoxPR_curve.png)

| Ground Truth | Predictions |
|---|---|
| ![Val Labels 0](backend/runs/detect/val/val_batch0_labels.jpg) | ![Val Preds 0](backend/runs/detect/val/val_batch0_pred.jpg) |
| ![Val Labels 1](backend/runs/detect/val/val_batch1_labels.jpg) | ![Val Preds 1](backend/runs/detect/val/val_batch1_pred.jpg) |

---

## 5. Stage 2 — Freshness Classification (ResNet50 Fine-tuning)

### 5.1 Training Configuration

| Parameter | Value |
|---|---|
| Base model | ResNet50 (ImageNet-1K V1 weights) |
| Dataset | 17-item fresh/rotten produce dataset (34 classes) |
| Input resize | 232 × 232 px |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| LR scheduler | Cosine annealing (T_max = 10) |
| Epochs | 10 |
| Backbone | Fully frozen (only classification head trained) |
| Augmentation | RandomResizedCrop, RandomHorizontalFlip, ColorJitter |
| Hardware | RTX 4060, CUDA |

### 5.2 Before vs. After Fine-tuning

| State | Validation Accuracy | Notes |
|---|---|---|
| ImageNet pre-trained only | ~0% on fresh/rotten task | No freshness classes exist in ImageNet |
| After 1 epoch | 82.96% | ImageNet features transfer extremely well |
| After 5 epochs | 91.07% | Learning rate still decreasing |
| **Best (Epoch 9)** | **91.52%** | Saved to `freshguard_resnet50.pth` |
| Final (Epoch 10) | 90.90% | Slight overfit on last step |

Training accuracy at best epoch: **82.58%** (gap vs. val accuracy suggests classification head is well-regularised but backbone features may benefit from unfreezing).

**Epoch-by-epoch breakdown:**

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 57.62% | 82.96% | 0.7544 |
| 2 | 75.55% | 86.95% | 0.5064 |
| 3 | 78.91% | 88.68% | 0.4101 |
| 4 | 80.10% | 89.40% | 0.3770 |
| 5 | 81.32% | 91.07% | 0.3329 |
| 6 | 81.59% | 90.42% | 0.3307 |
| 7 | 82.22% | 91.07% | 0.3127 |
| 8 | 82.45% | 90.94% | 0.3152 |
| **9** | **82.58%** | **91.52%** | **0.2999** |
| 10 | 82.76% | 90.90% | 0.3118 |

**Training curves:**

![ResNet50 Training History](backend/runs/train/cnn_finetune/images/resnet50.png)

---

## 6. CNN Model Comparison

Four architectures were evaluated to select the best classifier for production. All models shared the same training setup (10 epochs, AdamW, cosine annealing, frozen backbone, ImageNet weights), differing only in architecture, learning rate, and batch size.

**Summary table:**

| Model | Parameters | LR | Best Val Acc | Best Epoch | Final Val Acc |
|---|---|---|---|---|---|
| ResNet18 | ~11M | 1e-4 | 82.22% | 8 | 82.17% |
| MobileNet_v2 | ~3.4M | 3e-4 | 91.01% | 9 | 90.89% |
| EfficientNet_b0 | ~5.3M | 3e-4 | 89.62% | 9 | 89.41% |
| **ResNet50** | **~25M** | **1e-4** | **91.52%** | **9** | **90.90%** |

---

### 6.1 ResNet18

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 17.76% | 36.52% | 2.4057 |
| 2 | 40.65% | 59.16% | 1.7915 |
| 3 | 55.71% | 71.52% | 1.4057 |
| 4 | 63.68% | 76.21% | 1.1782 |
| 5 | 67.56% | 79.15% | 1.0472 |
| 6 | 69.90% | 80.46% | 0.9593 |
| 7 | 71.49% | 81.57% | 0.9101 |
| **8** | **72.21%** | **82.22%** | **0.8797** |
| 9 | 72.53% | 82.22% | 0.8697 |
| 10 | 72.87% | 82.17% | 0.8681 |

ResNet18 is the weakest of the four — its smaller representational capacity (11M params, shallow residual blocks) limits peak accuracy to 82.22%. The large gap between training accuracy (~72%) and validation accuracy (~82%) reflects the ImageNet backbone transferring better-than-expected features, but the head alone cannot extract finer freshness cues.

![ResNet18 Training History](backend/runs/train/cnn_finetune/images/resnet18.png)

---

### 6.2 MobileNet_v2

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 49.46% | 80.80% | 0.9899 |
| 2 | 75.28% | 86.75% | 0.6037 |
| 3 | 79.24% | 88.59% | 0.4807 |
| 4 | 80.79% | 89.56% | 0.4205 |
| 5 | 81.81% | 90.28% | 0.3868 |
| 6 | 82.72% | 90.18% | 0.3727 |
| 7 | 82.96% | 90.75% | 0.3571 |
| 8 | 83.21% | 90.75% | 0.3526 |
| **9** | **83.59%** | **91.01%** | **0.3479** |
| 10 | 83.36% | 90.89% | 0.3468 |

MobileNet_v2 converges fastest — epoch 1 already achieves 80.8% validation accuracy because its depthwise-separable convolutions are highly efficient at capturing texture patterns relevant to freshness. Despite having only 3.4M parameters it nearly matches ResNet50, making it a strong candidate for edge or mobile deployment.

![MobileNet_v2 Training History](backend/runs/train/cnn_finetune/images/mobilenet_v2_training.png)

---

### 6.3 EfficientNet_b0

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 45.71% | 77.23% | 1.3327 |
| 2 | 69.97% | 84.63% | 0.8113 |
| 3 | 74.91% | 86.69% | 0.6295 |
| 4 | 76.84% | 87.78% | 0.5418 |
| 5 | 78.21% | 88.50% | 0.4933 |
| 6 | 78.69% | 89.18% | 0.4598 |
| 7 | 79.28% | 89.32% | 0.4445 |
| 8 | 79.67% | 89.29% | 0.4353 |
| **9** | **80.02%** | **89.62%** | **0.4257** |
| 10 | 79.96% | 89.41% | 0.4273 |

EfficientNet_b0 lands between MobileNet and ResNet50 — compound scaling provides a good accuracy-to-size ratio (89.62% at only 5.3M params) but its validation loss never drops as low as the other two top performers, suggesting it benefits more from a higher learning rate during head training.

![EfficientNet_b0 Training History](backend/runs/train/cnn_finetune/images/efficientnet_b0.png)

---

### 6.4 ResNet50 (Selected for Production)

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 57.62% | 82.96% | 0.7544 |
| 2 | 75.55% | 86.95% | 0.5064 |
| 3 | 78.91% | 88.68% | 0.4101 |
| 4 | 80.10% | 89.40% | 0.3770 |
| 5 | 81.32% | 91.07% | 0.3329 |
| 6 | 81.59% | 90.42% | 0.3307 |
| 7 | 82.22% | 91.07% | 0.3127 |
| 8 | 82.45% | 90.94% | 0.3152 |
| **9** | **82.58%** | **91.52%** | **0.2999** |
| 10 | 82.76% | 90.90% | 0.3118 |

ResNet50 achieves the lowest validation loss (0.2999) and highest peak accuracy (91.52%). Its deeper residual blocks extract richer spatial and texture features — critical for distinguishing subtle visual differences between fresh and rotten produce (discolouration, surface texture degradation, shrinkage). Saved to `models/freshguard_resnet50.pth`.

![ResNet50 Training History](backend/runs/train/cnn_finetune/images/resnet50.png)

---

### 6.5 Head-to-Head Comparison

The plot below overlays all four models' validation loss and accuracy trajectories on the same axes:

![All Models Comparison](backend/runs/train/cnn_finetune/images/all_models_comparision.png)

**Key takeaways:**
- ResNet18 is clearly outpaced — not suitable for this task at 10 epochs
- MobileNet_v2, EfficientNet_b0, and ResNet50 are tightly clustered above 89% from epoch 4 onward
- ResNet50 edges ahead consistently in the second half of training, justifying its selection despite the larger parameter count
- All three top models plateau by epoch 9, suggesting more epochs alone would not yield major gains without unfreezing backbone layers

---

## 7. Inference Pipeline

```
1. User POSTs an image to  POST /detect

2. YOLO11m runs inference at confidence ≥ 0.30
   → Returns bounding boxes + class labels for detected produce

3. Each detection is cropped to a 480 px canvas (preserving aspect ratio)
   and saved as a temporary PNG

4. ResNet50 classifies each crop
   → Outputs one of 34 labels + softmax confidence

5. API assembles and returns:
   {
     "image_id": "<uuid>",
     "results": [
       { "file": "<crop>.png", "label": "FreshMango", "confidence": 0.94 },
       { "file": "<crop>.png", "label": "RottenTomato", "confidence": 0.87 }
     ]
   }

6. Temporary files (original upload, annotated image, crops) are deleted

7. Frontend parses labels → colour-coded badges
   Green  = fresh
   Red    = rotten
   Orange = overripe
   Amber  = stale / spoiled
```

---

## 8. Frontend

The Next.js frontend provides a single-page drag-and-drop interface:

- **Upload:** Drag-and-drop or click-to-browse; supports any browser-supported image format
- **Preview:** Instant thumbnail before submission
- **Loading state:** Spinner overlay while the API processes the image
- **Results panel:** One `DetectionCard` per detected item, showing:
  - Produce name (parsed from CamelCase or underscore label)
  - Freshness condition badge (colour-coded)
  - Confidence progress bar
- **Summary badges:** Total items detected, fresh count, rotten count
- **Error handling:** Inline error message with retry button
- **API target:** `http://localhost:8000/detect` (configurable)

---

## 9. Limitations

### 9.1 The Classification Gap — Produce Outside the 34 Classes

This is the most significant practical limitation. The **YOLO detector** can locate 63 different produce types. The **ResNet50 classifier** can only assign freshness to 17 of them (34 classes = 17 items × fresh + rotten). The remaining 46 detected produce items have **no matching classifier class**.

When YOLO detects, for example, an `avocado`, `pineapple`, `mushroom`, or `watermelon`, the classifier still runs — it will force-assign the nearest class from its 34-class vocabulary based on visual similarity. This means:

- A ripe **avocado** might be labelled `FreshMango` (green, similar skin texture)
- A **pineapple** might be labelled `FreshGuava` (yellow, bumpy surface)
- A **watermelon** slice might score as `freshcucumber` (green rind)
- Any produce item far from the training distribution will produce **unreliable confidence scores**

There is no "unknown" or "out-of-distribution" output path — the model always picks one of its 34 classes. Users should treat classifications of produce not listed in Section 3.3 with caution.

### 9.2 YOLO mAP Ceiling

A best mAP@50 of **0.275** (27.5%) is moderate. This means roughly 72.5% of ground-truth bounding boxes are either missed or incorrectly placed at IoU@50 threshold. Contributing factors:

- **63-class long tail:** Several produce classes (e.g., `date`, `prune`, `persimmon`, `gourd`) are visually similar and share limited training samples in the LVIS dataset
- **Duplicate class IDs** in the dataset annotations (strawberry at IDs 30 & 57; tomato at 35, 55 & 59) introduce label noise
- **Training was stopped early** at epoch 45 (best at epoch 25); longer training with unfrozen backbone layers may improve mAP
- **Small produce items** (chickpea, blueberry, pea) are difficult to detect at 640 px resolution

### 9.3 ResNet50 Training Accuracy Gap

Best validation accuracy is **91.52%** but training accuracy peaks at **82.76%** — an unusual inverse gap (val > train). This occurs because:
- The backbone is fully frozen: only the linear head is trained
- ImageNet features generalise well to this domain, so validation sees strong representations
- The classification head has not fully saturated — unfreezing the last few backbone blocks would likely push both train and val accuracy higher

### 9.4 Confidence Threshold Sensitivity

The YOLO confidence threshold is fixed at **0.30**. At this level:
- Low-quality or partially-occluded produce may be missed
- Visually ambiguous objects (a green ball near vegetables) may be falsely detected as produce
- Increasing the threshold reduces false positives at the cost of more missed detections

### 9.5 Model Is Not Production-Perfect

The system is a proof-of-concept. Real-world deployment would require:
- Expanding the classification dataset to cover all 63 detectable classes (not just 17)
- Retraining YOLO with more epochs and a partially unfrozen backbone
- Adding an out-of-distribution detector to flag produce not in the classifier's vocabulary
- Handling edge cases: mixed produce in one frame, heavily occluded items, non-standard lighting conditions
- HTTPS + authentication for the API endpoint (currently localhost only)

---

