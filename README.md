# Brain Tumor Detector (v2)

Binary image classifier (tumor vs no-tumor) using a simple folder dataset:

```
brain_tumor_dataset/
  yes/   # tumor images
  no/    # non-tumor images
  all/   # optional; ignored
```

## What this repo contains

- `brain_tumor_detector_v2.py`: baseline trainer + evaluation script
- Supports:
  - **Custom CNN** on grayscale images
  - **MobileNetV2 transfer learning** (recommended baseline)
  - Optional **two-phase fine-tuning** for MobileNetV2
  - **Threshold policies** (balanced accuracy or sensitivity-first “hospital-style”)
  - Saved plots: training curves + ROC + Precision/Recall

> Not a medical device. This is an educational ML project. Do not use for clinical decisions.

## Setup

Create a venv and install deps:

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -U pip
pip install tensorflow pillow numpy matplotlib scikit-learn
```

## Usage

### 1) Baseline (custom CNN)

```bash
python brain_tumor_detector_v2.py --save-plots
```

### 2) Transfer learning (MobileNetV2)

```bash
python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --save-plots
```

MobileNetV2 automatically switches image size to **224x224** unless you override `--img-h/--img-w`.

### 3) “Hospital-style” thresholding (sensitivity-first)

Chooses a threshold on the validation set that **keeps recall(class=1) >= target**, and then maximizes specificity.

```bash
python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --save-plots \
  --threshold-policy min_recall --target-recall 0.95 --monitor val_pr_auc
```

### 4) Two-phase fine-tune (MobileNetV2)

```bash
python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --fine-tune \
  --head-epochs 12 --ft-epochs 20 --ft-lr 1e-4 --unfreeze-last 40 \
  --monitor val_pr_auc --save-plots
```

## What gets printed

- Dataset split sizes + class balance
- Model summary
- Threshold-free evaluation: **loss / accuracy (0.50 threshold) / ROC-AUC / PR-AUC**
- Confusion matrices and metrics @ **0.50** and @ **chosen threshold**
  - acc
  - sensitivity / recall for tumor class (class 1)
  - specificity / recall for no-tumor class (class 0)
- Classification report at chosen threshold

## Common next upgrades

- Swap to a larger dataset (e.g., a Kaggle “brain tumor MRI/CT” dataset) and re-run the same script
- Add k-fold cross validation or repeated splits (small datasets are high-variance)
- Calibrate probabilities (temperature scaling) before locking a clinical-style threshold
- Add Grad-CAM / saliency maps for qualitative sanity checks
