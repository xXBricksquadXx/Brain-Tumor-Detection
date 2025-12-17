"""
brain_tumor_detector_v2.py

Binary classifier (tumor vs no-tumor) from folder structure:

  brain_tumor_dataset/
    yes/
    no/
    all/   (optional; ignored)

Run:
  python brain_tumor_detector_v2.py
  python brain_tumor_detector_v2.py --save-plots
  python brain_tumor_detector_v2.py --plots
  python brain_tumor_detector_v2.py --val-size 0.30 --save-plots
  python brain_tumor_detector_v2.py --threshold 0.60 --save-plots

Notes:
- Default val-size is set to 0.30 (larger val split for more stable thresholding).
- Imports avoid `from tensorflow.keras import ...` to reduce Pylance resolution issues.
- If --threshold is omitted, threshold is picked on validation set (balanced accuracy),
  skipping degenerate thresholds (all-0 / all-1).
- --save-plots writes accuracy.png / auc.png / loss.png to the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

import tensorflow as tf
from tensorflow import keras


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./brain_tumor_dataset",
                   help="Dataset root containing yes/ and no/")
    p.add_argument("--img-h", type=int, default=135, help="Image height")
    p.add_argument("--img-w", type=int, default=240, help="Image width")

    p.add_argument("--test-size", type=float, default=0.33, help="Test split fraction")
    # Larger default validation split for stability
    p.add_argument("--val-size", type=float, default=0.30, help="Val fraction of TRAIN split")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--epochs", type=int, default=50, help="Max epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    p.add_argument("--model-out", type=str, default="model.keras", help="Path to save best model")

    p.add_argument("--plots", action="store_true", help="Show plots (GUI)")
    p.add_argument("--save-plots", "--save-plot", dest="save_plots", action="store_true",
                   help="Save plots to PNG files (accuracy.png, auc.png, loss.png)")

    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold for class=1. If omitted, pick from validation set.")
    p.add_argument("--use-class-weights", action="store_true",
                   help="Apply class weights computed from training labels")

    return p.parse_args()


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in sorted(dir_path.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


def load_images(paths: List[Path], label: int, img_h: int, img_w: int) -> Tuple[List[np.ndarray], List[int]]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for p in paths:
        img = Image.open(p).convert("L")
        img = ImageOps.pad(img, (img_w, img_h), color=0)  # deterministic, preserves aspect
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X.append(arr)
        y.append(label)

    return X, y


def make_splits(data_dir: Path, img_h: int, img_w: int, test_size: float, val_size: float, seed: int):
    yes_dir = data_dir / "yes"
    no_dir = data_dir / "no"

    assert yes_dir.exists(), f"Missing folder: {yes_dir.resolve()}"
    assert no_dir.exists(), f"Missing folder: {no_dir.resolve()}"

    yes_paths = list_images(yes_dir)
    no_paths = list_images(no_dir)

    assert len(yes_paths) > 0, f"No images found in: {yes_dir.resolve()}"
    assert len(no_paths) > 0, f"No images found in: {no_dir.resolve()}"

    X_yes, y_yes = load_images(yes_paths, 1, img_h, img_w)
    X_no, y_no = load_images(no_paths, 0, img_h, img_w)

    X = np.array(X_yes + X_no, dtype=np.float32)[..., None]  # (N,H,W,1)
    y = np.array(y_yes + y_no, dtype=np.float32)            # (N,)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Train/val split from train (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(img_h: int, img_w: int, lr: float) -> keras.Model:
    # Use keras.layers to avoid `from tensorflow.keras import layers`
    L = keras.layers

    model = keras.Sequential([
        keras.Input(shape=(img_h, img_w, 1)),
        L.Conv2D(16, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Conv2D(32, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Conv2D(64, 3, activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dropout(0.30),
        L.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def pick_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Pick threshold maximizing balanced accuracy on validation set.
    Skips degenerate thresholds that predict a single class for all samples.
    Falls back to 0.5 if the model can't separate classes on validation.
    """
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        if preds.min() == preds.max():  # all 0s or all 1s
            continue
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score, best_t = score, float(t)

    if best_score < 0:
        return 0.5
    return best_t


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    # weights inversely proportional to class frequency
    y = y_train.astype(int)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    total = n0 + n1
    w0 = total / (2 * max(n0, 1))
    w1 = total / (2 * max(n1, 1))
    return {0: float(w0), 1: float(w1)}


def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(
        data_dir=data_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Pos rate (train): {y_train.mean():.3f} | (val): {y_val.mean():.3f} | (test): {y_test.mean():.3f}")

    model = build_model(args.img_h, args.img_w, args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_out, monitor="val_auc", mode="max", save_best_only=True),
    ]

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_train)
        print(f"Using class weights: {class_weight}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")
    print(f"Test auc : {test_auc:.4f}")
    print(f"Saved best model to: {Path(args.model_out).resolve()}")

    # --- Threshold selection (val) or user provided ---
    val_probs = model.predict(X_val, verbose=0).reshape(-1)
    print("\nVal prob stats:", f"min={val_probs.min():.3f} mean={val_probs.mean():.3f} max={val_probs.max():.3f}")

    if args.threshold is None:
        thr = pick_threshold(y_val.astype(int), val_probs)
        print(f"Picked threshold from val: {thr:.2f}")
    else:
        thr = float(args.threshold)
        print(f"Using user threshold: {thr:.2f}")

    # --- Validation confusion at BOTH thresholds ---
    val_preds_05 = (val_probs >= 0.5).astype(int)
    val_preds_thr = (val_probs >= thr).astype(int)

    print("\nVal Confusion Matrix @0.50 (rows=true, cols=pred):")
    print(confusion_matrix(y_val.astype(int), val_preds_05))

    print(f"\nVal Confusion Matrix @{thr:.2f} (rows=true, cols=pred):")
    print(confusion_matrix(y_val.astype(int), val_preds_thr))

    # --- Test predictions ---
    test_probs = model.predict(X_test, verbose=0).reshape(-1)
    print("Test prob stats:", f"min={test_probs.min():.3f} mean={test_probs.mean():.3f} max={test_probs.max():.3f}")

    test_preds_05 = (test_probs >= 0.5).astype(int)
    test_preds_thr = (test_probs >= thr).astype(int)

    print("\nTest Confusion Matrix @0.50 (rows=true, cols=pred):")
    print(confusion_matrix(y_test.astype(int), test_preds_05))

    print(f"\nTest Confusion Matrix @{thr:.2f} (rows=true, cols=pred):")
    print(confusion_matrix(y_test.astype(int), test_preds_thr))

    print(f"\nClassification Report @{thr:.2f}:")
    print(classification_report(y_test.astype(int), test_preds_thr, digits=4, zero_division=0))

    # --- Plots ---
    def _maybe_save_or_show(fig_name: str):
        if args.save_plots:
            plt.savefig(fig_name, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if args.plots or args.save_plots:
        # Accuracy
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        _maybe_save_or_show("accuracy.png")

        # AUC
        plt.plot(history.history["auc"], label="train")
        plt.plot(history.history["val_auc"], label="val")
        plt.title("AUC")
        plt.xlabel("epoch")
        plt.ylabel("auc")
        plt.legend()
        _maybe_save_or_show("auc.png")

        # Loss
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        _maybe_save_or_show("loss.png")


if __name__ == "__main__":
    main()
