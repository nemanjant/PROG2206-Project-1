# Plant Segmentation & Trait Analysis (Prototype)

**Status:** Prototype / learning project (first attempt at ML + computer vision)

This repository explores a simple pipeline for **plant structure segmentation** (mask proposals) and **basic trait extraction** (mask features that can support leaf/stem classification).

> **Important limitation (read this first):**  
> The initial experiment behind this repo used **only one plant image** as training data. That means there is **no meaningful train/validation/test split**, and any “accuracy” you might observe is **not statistically valid**.  
> Treat this as a **pipeline prototype** and a learning artifact—not a finished ML model.

---

## What this repo demonstrates

- Using **Segment Anything (SAM)** to generate multiple candidate masks for a plant image
- Creating a small labeled dataset of masks (`leaf` vs `stem`) via `data/labels.csv`
- Extracting simple mask-level features (shape / geometry / basic image statistics)
- Training a baseline classifier on those features (e.g., RandomForest)
- Running inference and exporting mask features + visual overlay

---

## Repository structure

```
data/
  raw_images/                 # input images
  masks/                      # generated SAM masks (binary PNGs)
  labels.csv                  # manual labels for masks
  labels_with_features.csv    # labels + extracted features
  results/                    # inference outputs (CSV)
models/
  sam/                        # SAM checkpoint goes here (NOT committed)
  classifier/                 # trained classifier artifact(s)
scripts/
  generate_masks.py
  train_classifier.py
  inference.py
  utils/
    dataset.py                # feature extraction for labeled masks
    transform.py
```

---

## Setup

### 1) Create a virtual environment

**Windows (PowerShell):**
```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2) Install dependencies

Your current `requirements.txt` is minimal/empty, so either:
- install manually, then freeze later; or
- populate `requirements.txt` and install from it.

Common packages used in this type of pipeline:
```bash
pip install numpy pandas opencv-python matplotlib scikit-image scikit-learn joblib
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> Tip: After everything works, generate a real requirements file:
```bash
pip freeze > requirements.txt
```

### 3) Download the SAM checkpoint

Download **`sam_vit_h_4b8939.pth`** and place it here:
```
models/sam/sam_vit_h_4b8939.pth
```

---

## Quickstart

### Step A — Add an input image
Put one or more images into:
```
data/raw_images/
```

### Step B — Generate masks using SAM
```bash
python scripts/generate_masks.py
```
Outputs:
- `data/masks/<image_name>_mask_<i>.png`

### Step C — Label a subset of masks
Open `data/masks/` and label masks in:
```
data/labels.csv
```

Format:
```csv
image_name,mask_name,type
plant_image.jpg,plant_image_mask_1.png,leaf
plant_image.jpg,plant_image_mask_2.png,stem
```

### Step D — Extract features for labeled masks
```bash
python scripts/utils/dataset.py
```
Outputs:
- `data/labels_with_features.csv`

### Step E — Train a baseline classifier
```bash
python scripts/train_classifier.py
```
Outputs:
- a classifier artifact saved under `models/classifier/` (name depends on script)

### Step F — Run inference
```bash
python scripts/inference.py
```
Outputs:
- `data/results/<image>_features.csv`
- visualization overlay (window or saved output, depending on script)

---

## Known limitations

- **Dataset size (critical):** one image is not enough for real evaluation.
- **Label quality:** manual mask labeling is subjective and time-consuming.
- **Generalization:** model likely overfits to lighting/background of the single image.

---
