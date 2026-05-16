# FreshGuard — Project Handover

## Goal

A web application that lets a user upload a photo containing one or more fruits or
vegetables and returns which items are present and whether each is **fresh or rotten**.

---

## Architecture

```
frontend (Next.js)
    │  POST /detect  (multipart file upload)
    ▼
backend (FastAPI — main.py)
    │
    ▼
detection.py  ─── finetuned YOLO (models/freshguard_yolo/weights/best.pt)
                   detects bounding boxes + classifies Fresh/Rotten in one pass
```

**34 classes** — 17 produce types × Fresh/Rotten:
FreshApple, rottenapples, freshbanana, rottenbanana, FreshBellpepper, RottenBellpepper,
freshbittergroud, rottenbittergroud, freshcapsicum, rottencapsicum, FreshCarrot, RottenCarrot,
freshcucumber, rottencucumber, FreshGrape, RottenGrape, FreshGuava, RottenGuava,
FreshJujube, RottenJujube, FreshMango, RottenMango, freshokra, rottenokra,
FreshOrange, rottenoranges, FreshPomegranate, RottenPomegranate, freshpotato, rottenpotato,
FreshStrawberry, RottenStrawberry, freshtomato, rottentomato

---

## Current State

### Data  (`backend/data/`)
| Split | Images | Location |
|-------|--------|----------|
| Train | ~37 K  | `data/train/<class>/` |
| Val   | ~10 K  | `data/val/<class>/` |

Split was created by moving 20 % of each class from train → val (random seed 42).

### Models  (`backend/models/`)
**Empty — no model has been trained yet.**

Two models need to be produced before the app runs end-to-end:

| Model | Purpose | Saved to |
|-------|---------|----------|
| `freshguard_resnet50.pth` | Freshness classifier (ResNet-50, 34 classes) | `models/` |
| `freshguard_yolo/weights/best.pt` | Detection + classification (YOLO, 34 classes) | `models/freshguard_yolo/` |

`detection.py` currently loads **only the YOLO model**. The ResNet-50 model is no longer
used at runtime — it was an intermediate step before the YOLO pipeline was designed.

### Notebook  (`backend/finetune.ipynb`)

| Cells | What they do | Status |
|-------|-------------|--------|
| 1–4   | Imports, device, seed | Ready |
| 5–8   | Transforms, datasets (`data/train` + `data/val`), DataLoaders | Ready |
| 9–13  | Training loop, eval loop, plotting helpers | Ready |
| 14–17 | ResNet-50 build, finetune, train | **Not yet run** |
| 18    | Export ResNet-50 weights + `class_names.json` to `models/` | **Not yet run** |
| 19–21 | YOLO dataset prep + YOLO finetune | **Not yet run** |

### Backend  (`backend/`)
| File | Status |
|------|--------|
| `main.py` | Working FastAPI app — exposes `POST /detect` |
| `detection.py` | Complete — loads finetuned YOLO, runs detection pipeline |
| `prepare_yolo_dataset.py` | Complete — run once before YOLO training |
| `finetune.ipynb` | Complete — full training pipeline ready to execute |

### Frontend  (`frontend/`)
Basic Next.js UI — upload button, detect button, results list. Has **two bugs** that
need fixing before it connects to the backend correctly (see Next Steps below).

---

## Next Steps  (in order)

### 1. Train ResNet-50 classifier
Run cells 1–18 in `finetune.ipynb`.
Produces `models/freshguard_resnet50.pth` and `models/class_names.json`.
These are not needed at runtime but are useful as a standalone freshness classifier.

### 2. Prepare YOLO dataset
```bash
cd backend
python prepare_yolo_dataset.py
```
Creates `data/yolo/` with image junctions, per-image label files (`class_id 0.5 0.5 1.0 1.0`),
and `data/yolo/dataset.yaml`. Takes a few minutes — writes ~47 K label files.

### 3. Train YOLO detector
Run cells 19–21 in `finetune.ipynb`.
Finetunes `yolo11x.pt` on all 34 classes for 50 epochs (~2–4 hours on RTX 4060).
Best weights are saved to `models/freshguard_yolo/weights/best.pt`.
`detection.py` is already wired to load from that path.

### 4. Fix frontend bugs

**Bug 1 — wrong form field name** ✅ Fixed
`page.tsx` was sending the file under key `'image'`; FastAPI expects `'file'`.
FastAPI would have returned a 422 for every request. Fixed in `page.tsx` line 34.

**Bug 2 — confidence display** ✅ No change needed
Backend returns confidence as a `0–1` float. Frontend does `Math.round(conf * 100)%`
which is correct (e.g. `0.94` → `94%`).

**Bug 3 — API port mismatch** ⚠️ Start the server on the right port
Frontend calls `http://localhost:5000/detect`. FastAPI defaults to 8000.
Start the backend on port 5000 to match:
```bash
uvicorn main:app --port 5000 --reload
```

**Bug 4 — stale `Detection` interface** ✅ Fixed
`bbox: number[]` removed, `path: string` added to match the actual response shape.

### 5. End-to-end test
Start the backend:
```bash
cd backend
uvicorn main:app --port 5000 --reload
```
Start the frontend:
```bash
cd frontend
npm run dev
```
Upload a photo with multiple produce items and verify detections come back correctly.
