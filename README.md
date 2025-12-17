# Brain Tumor Detector (v2)

Educational / research prototype for **binary classification** of brain scan images:

- `1` = tumor (folder: `yes/`)
- `0` = no tumor (folder: `no/`)

This repo is **not medical advice** and is **not validated for clinical use**.

## What’s new in v2

- Stable splitting and evaluation:
  - stratified train/val/test split
  - **larger default val split** (`--val-size 0.30`) to reduce noisy threshold tuning
- Cleaner training/eval script (`brain_tumor_detector_v2.py`)
  - evaluates confusion matrices at **0.50** and at the **chosen threshold**
  - if `--threshold` is omitted, auto-picks a threshold using validation balanced accuracy
- Plot outputs
  - `accuracy.png`, `auc.png`, `loss.png`
  - `roc.png`, `pr.png`

## Dataset layout

Place your dataset here (not committed to git):

**brain_tumor_dataset/**

- yes/
  _.png / _.jpg / ...
- no/
  _.png / _.jpg / ...

  > Notes:

- Images are loaded as grayscale and padded/resized deterministically.
- `all/` is ignored if present.

## Quickstart (Windows PowerShell)

Create venv + install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install tensorflow scikit-learn matplotlib pillow
```

> Run baseline + save plots:

```
python brain_tumor_detector_v2.py --save-plots
```

> Run with a specific decision threshold:

```
python brain_tumor_detector_v2.py --threshold 0.60 --save-plots
```
