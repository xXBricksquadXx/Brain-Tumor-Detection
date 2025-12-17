"""
brain_tumor_detector_v2.py

Binary classifier (tumor vs no-tumor) using folder structure:

  brain_tumor_dataset/
    yes/
    no/
    all/   (optional; ignored)

Key features:
- Stratified train/val/test split with larger default val split (val-size=0.30 of TRAIN split).
- Pylance-friendly imports (avoid `from tensorflow.keras import ...`).
- Custom CNN (grayscale) OR MobileNetV2 transfer learning (expects 224x224x3).
- Optional two-phase MobileNetV2 fine-tune:
  Phase 1: frozen backbone (train head)
  Phase 2: unfreeze last N layers (keep BatchNorm frozen) + lower LR
- Thresholding:
  - Always prints confusion matrices @0.50 and @chosen threshold
  - If --threshold is omitted, chooses threshold from validation set using:
      * balanced_accuracy  (default)
      * min_recall (sensitivity-first; pick max specificity while recall>=target)
- Adds PR-AUC metric (threshold-free) for monitoring and evaluation.
- Saves training curves + ROC + PR curves when --save-plots is used.

Examples:
  python brain_tumor_detector_v2.py --save-plots

  python brain_tumor_detector_v2.py --threshold 0.60 --save-plots

  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --save-plots

  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment \
    --threshold-policy min_recall --target-recall 0.95 --monitor val_pr_auc --save-plots

  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --fine-tune \
    --head-epochs 12 --ft-epochs 20 --ft-lr 1e-4 --unfreeze-last 40 --monitor val_pr_auc --save-plots

NOTE: This is a learning project and is NOT a medical device. Do not use for clinical decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

import tensorflow as tf
from tensorflow import keras


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="./brain_tumor_dataset",
        help="Dataset root containing yes/ and no/",
    )

    p.add_argument("--img-h", type=int, default=135, help="Image height")
    p.add_argument("--img-w", type=int, default=240, help="Image width")

    p.add_argument("--test-size", type=float, default=0.33, help="Test split fraction")
    p.add_argument(
        "--val-size",
        type=float,
        default=0.30,
        help="Val fraction of TRAIN split (after test split)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Single-stage training (custom CNN or frozen TL if not using fine-tune)
    p.add_argument("--epochs", type=int, default=50, help="Max epochs (single-stage mode)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (single-stage OR head stage)")

    # Two-phase fine-tune options (MobileNetV2 only)
    p.add_argument(
        "--fine-tune",
        action="store_true",
        help="Enable two-phase fine-tuning (MobileNetV2 backbone only)",
    )
    p.add_argument("--head-epochs", type=int, default=12, help="Phase 1 epochs (frozen backbone; head training)")
    p.add_argument("--ft-epochs", type=int, default=20, help="Phase 2 epochs (partial unfreeze; fine-tuning)")
    p.add_argument("--ft-lr", type=float, default=1e-4, help="Phase 2 learning rate (fine-tuning)")
    p.add_argument(
        "--unfreeze-last",
        type=int,
        default=40,
        help="How many backbone layers to unfreeze in phase 2 (BatchNorm stays frozen)",
    )

    p.add_argument("--model-out", type=str, default="model.keras", help="Path to save best model")

    p.add_argument("--plots", action="store_true", help="Show plots (GUI)")
    p.add_argument(
        "--save-plots",
        "--save-plot",
        dest="save_plots",
        action="store_true",
        help="Save plots to PNG files in the repo (accuracy/auc/pr_auc/loss/roc/pr)",
    )

    # Threshold selection
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for class=1. If omitted, choose from validation set (per --threshold-policy).",
    )
    p.add_argument(
        "--threshold-policy",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "min_recall"],
        help="How to pick threshold from validation set when --threshold is omitted.",
    )
    p.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Used when --threshold-policy=min_recall (sensitivity-first).",
    )

    p.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply class weights computed from training labels (often not needed here).",
    )

    # Backbone
    p.add_argument(
        "--backbone",
        type=str,
        default="custom",
        choices=["custom", "mobilenetv2"],
        help="Model backbone: custom (grayscale CNN) or mobilenetv2 (transfer learning).",
    )
    p.add_argument(
        "--backbone-weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="Backbone weights. 'imagenet' downloads pretrained weights; 'none' initializes randomly.",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable lightweight augmentation (recommended for transfer learning).",
    )

    # Callback monitoring
    p.add_argument(
        "--monitor",
        type=str,
        default="val_auc",
        help="Metric to monitor for EarlyStopping/ModelCheckpoint (e.g. val_auc, val_pr_auc, val_loss).",
    )
    p.add_argument("--patience", type=int, default=7, help="EarlyStopping patience (single-stage)")
    p.add_argument("--patience-ft", type=int, default=5, help="EarlyStopping patience (fine-tune phase)")

    return p.parse_args()


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def load_images(paths: List[Path], label: int, img_h: int, img_w: int) -> Tuple[List[np.ndarray], List[int]]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for p in paths:
        try:
            img = Image.open(p).convert("L")
            img = ImageOps.pad(img, (img_w, img_h), color=0)  # deterministic, preserves aspect
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
            X.append(arr)
            y.append(label)
        except Exception as e:
            print(f"WARNING: failed to load image: {p} ({e})")

    return X, y


def make_splits(
    data_dir: Path,
    img_h: int,
    img_w: int,
    channels: int,
    test_size: float,
    val_size: float,
    seed: int,
):
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

    X = np.array(X_yes + X_no, dtype=np.float32)  # (N,H,W)
    y = np.array(y_yes + y_no, dtype=np.float32)  # (N,)

    X = X[..., None]  # (N,H,W,1)
    if channels == 3:
        X = np.repeat(X, 3, axis=-1)  # (N,H,W,3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compile_binary(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_model_custom(img_h: int, img_w: int, lr: float, augment: bool) -> keras.Model:
    L = keras.layers

    aug = (
        keras.Sequential(
            [
                L.RandomRotation(0.03),
                L.RandomZoom(0.08),
                L.RandomContrast(0.10),
            ],
            name="augment",
        )
        if augment
        else None
    )

    inputs = keras.Input(shape=(img_h, img_w, 1))
    x = inputs
    if aug is not None:
        x = aug(x)

    x = L.Conv2D(16, 3, activation="relu")(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(32, 3, activation="relu")(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(64, 3, activation="relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.30)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="custom_cnn")
    compile_binary(model, lr)
    return model


def build_model_mobilenetv2(
    img_h: int,
    img_w: int,
    lr: float,
    weights: str,
    augment: bool,
) -> Tuple[keras.Model, keras.Model]:
    """
    Returns: (full_model, backbone_model)
    """
    L = keras.layers

    aug = (
        keras.Sequential(
            [
                L.RandomRotation(0.03),
                L.RandomZoom(0.10),
                L.RandomContrast(0.12),
            ],
            name="augment",
        )
        if augment
        else None
    )

    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = inputs

    # MobileNetV2 expects inputs roughly in [-1, 1]
    x = L.Rescaling(2.0, offset=-1.0)(x)

    if aug is not None:
        x = aug(x)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights=(None if weights == "none" else "imagenet"),
        input_shape=(img_h, img_w, 3),
        pooling="avg",
    )
    backbone.trainable = False  # phase 1: frozen

    x = backbone(x)
    x = L.Dropout(0.35)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_head")
    compile_binary(model, lr)
    return model, backbone


def unfreeze_backbone_for_finetune(backbone: keras.Model, unfreeze_last: int) -> None:
    """
    Unfreeze last N layers of backbone for fine-tuning, but keep BatchNorm layers frozen.
    """
    backbone.trainable = True

    if unfreeze_last is not None and unfreeze_last > 0:
        for layer in backbone.layers[:-unfreeze_last]:
            layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    y = y_train.astype(int)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    total = n0 + n1
    w0 = total / (2 * max(n0, 1))
    w1 = total / (2 * max(n1, 1))
    return {0: float(w0), 1: float(w1)}


def _safe_rates(cm: np.ndarray) -> Tuple[float, float, float]:
    # cm layout (rows=true, cols=pred): [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / max(tn + fp + fn + tp, 1)
    sens = tp / max(tp + fn, 1)  # recall for class 1
    spec = tn / max(tn + fp, 1)  # recall for class 0
    return float(acc), float(sens), float(spec)


def print_threshold_block(name: str, y_true: np.ndarray, probs: np.ndarray, thr: float) -> None:
    y_int = y_true.astype(int)
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(y_int, preds)
    acc, sens, spec = _safe_rates(cm)
    print(f"\n{name} Confusion Matrix @{thr:.2f} (rows=true, cols=pred):")
    print(cm)
    print(f"{name} metrics @{thr:.2f}: acc={acc:.4f} sens(recall1)={sens:.4f} spec(recall0)={spec:.4f}")


def pick_threshold_balanced_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        if preds.min() == preds.max():
            continue
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score, best_t = score, float(t)
    return 0.5 if best_score < 0 else best_t


def pick_threshold_min_recall(y_true: np.ndarray, probs: np.ndarray, target_recall: float) -> float:
    """
    Sensitivity-first thresholding:
    - Find thresholds where recall(class1) >= target_recall
    - Choose the one with maximum specificity (and then best balanced accuracy, then higher threshold)
    - Skip degenerate thresholds (all-0 or all-1 predictions)
    """
    y = y_true.astype(int)

    candidates: List[Tuple[float, float, float]] = []  # (spec, bal_acc, thr)
    fallback: List[Tuple[float, float, float, float]] = []  # (recall1, spec, bal_acc, thr)

    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        if preds.min() == preds.max():
            continue

        cm = confusion_matrix(y, preds)
        acc, sens, spec = _safe_rates(cm)
        bal = 0.5 * (sens + spec)

        fallback.append((sens, spec, bal, float(t)))
        if sens >= target_recall:
            candidates.append((spec, bal, float(t)))

    if candidates:
        # max specificity, then max bal, then higher threshold
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return candidates[0][2]

    # If nothing hits target recall, choose best available recall first, then specificity, then bal, then threshold.
    if fallback:
        fallback.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        return fallback[0][3]

    return 0.5


def monitor_mode(monitor: str) -> str:
    m = monitor.lower()
    return "min" if "loss" in m else "max"


def plot_and_maybe_save_or_show(fig_name: str, save_plots: bool) -> None:
    plt.tight_layout()
    if save_plots:
        plt.savefig(fig_name, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def merge_histories(h1: keras.callbacks.History, h2: Optional[keras.callbacks.History]) -> Dict[str, List[float]]:
    merged = {k: list(v) for k, v in h1.history.items()}
    if h2 is None:
        return merged
    for k, v in h2.history.items():
        merged.setdefault(k, [])
        merged[k].extend(list(v))
    return merged


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data_dir)

    channels = 1 if args.backbone == "custom" else 3
    if args.backbone != "custom":
        # If user didn't override defaults, auto-switch to 224
        if (args.img_h, args.img_w) == (135, 240):
            args.img_h, args.img_w = 224, 224
            print("NOTE: --backbone mobilenetv2 detected; auto-setting --img-h/--img-w to 224x224.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(
        data_dir=data_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        channels=channels,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Pos rate (train): {y_train.mean():.3f} | (val): {y_val.mean():.3f} | (test): {y_test.mean():.3f}")

    backbone = None
    if args.backbone == "custom":
        model = build_model_custom(args.img_h, args.img_w, args.lr, augment=args.augment)
    else:
        model, backbone = build_model_mobilenetv2(
            args.img_h,
            args.img_w,
            args.lr,
            weights=args.backbone_weights,
            augment=args.augment,
        )

    model.summary()

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_train)
        print(f"Using class weights: {class_weight}")

    mon = args.monitor
    mode = monitor_mode(mon)

    def make_callbacks(patience: int) -> List[keras.callbacks.Callback]:
        return [
            keras.callbacks.EarlyStopping(monitor=mon, mode=mode, patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(args.model_out, monitor=mon, mode=mode, save_best_only=True),
        ]

    # -------------------------
    # Training (single-stage OR two-phase)
    # -------------------------
    history2 = None

    if args.fine_tune and args.backbone == "mobilenetv2":
        # Phase 1
        print(f"\n[Phase 1] Frozen backbone head training: epochs={args.head_epochs}, lr={args.lr} monitor={mon}({mode})")
        history1 = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.head_epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(patience=max(2, args.patience)),
            class_weight=class_weight,
            verbose=1,
        )

        # Phase 2
        assert backbone is not None
        unfreeze_backbone_for_finetune(backbone, args.unfreeze_last)
        compile_binary(model, args.ft_lr)

        print(
            f"\n[Phase 2] Fine-tune: epochs={args.ft_epochs}, ft_lr={args.ft_lr}, "
            f"unfreeze_last={args.unfreeze_last} monitor={mon}({mode})"
        )
        history2 = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.ft_epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(patience=max(2, args.patience_ft)),
            class_weight=class_weight,
            verbose=1,
        )

        history = type("MergedHistory", (), {"history": merge_histories(history1, history2)})()
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(patience=max(2, args.patience)),
            class_weight=class_weight,
            verbose=1,
        )

    # -------------------------
    # Evaluation (threshold-free)
    # -------------------------
    eval_vals = model.evaluate(X_test, y_test, verbose=0)
    eval_map = {k: float(v) for k, v in zip(model.metrics_names, eval_vals)}

    print(f"\nTest loss: {eval_map.get('loss', float('nan')):.4f}")
    if "accuracy" in eval_map:
        print(f"Test acc : {eval_map['accuracy']:.4f}  (threshold=0.50)")
    if "auc" in eval_map:
        print(f"Test auc : {eval_map['auc']:.4f}")
    if "pr_auc" in eval_map:
        print(f"Test pr_auc: {eval_map['pr_auc']:.4f}")

    print(f"Saved best model to: {Path(args.model_out).resolve()}")

    # -------------------------
    # Threshold selection (validation)
    # -------------------------
    val_probs = model.predict(X_val, verbose=0).reshape(-1)
    print("\nVal prob stats:", f"min={val_probs.min():.3f} mean={val_probs.mean():.3f} max={val_probs.max():.3f}")

    if args.threshold is None:
        if args.threshold_policy == "balanced_accuracy":
            thr = pick_threshold_balanced_accuracy(y_val.astype(int), val_probs)
            print(f"Picked threshold from val (balanced_accuracy): {thr:.2f}")
        else:
            thr = pick_threshold_min_recall(y_val.astype(int), val_probs, target_recall=float(args.target_recall))
            print(f"Picked threshold from val (min_recall target={args.target_recall:.2f}): {thr:.2f}")
    else:
        thr = float(args.threshold)
        print(f"Using user threshold: {thr:.2f}")

    # Always print @0.50 and @thr for VAL + TEST
    print_threshold_block("Val", y_val, val_probs, 0.50)
    print_threshold_block("Val", y_val, val_probs, thr)

    test_probs = model.predict(X_test, verbose=0).reshape(-1)
    print("\nTest prob stats:", f"min={test_probs.min():.3f} mean={test_probs.mean():.3f} max={test_probs.max():.3f}")

    print_threshold_block("Test", y_test, test_probs, 0.50)
    print_threshold_block("Test", y_test, test_probs, thr)

    test_preds_thr = (test_probs >= thr).astype(int)
    print(f"\nClassification Report @{thr:.2f}:")
    print(classification_report(y_test.astype(int), test_preds_thr, digits=4, zero_division=0))

    # -------------------------
    # ROC + PR curves on TEST
    # -------------------------
    try:
        roc_auc = roc_auc_score(y_test.astype(int), test_probs)
        ap = average_precision_score(y_test.astype(int), test_probs)

        fpr, tpr, _ = roc_curve(y_test.astype(int), test_probs)
        prec, rec, _ = precision_recall_curve(y_test.astype(int), test_probs)

        print(f"\nROC-AUC (test): {roc_auc:.4f}")
        print(f"PR-AUC / Average Precision (test): {ap:.4f}")

        if args.plots or args.save_plots:
            plt.figure()
            plt.plot(fpr, tpr)
            plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plot_and_maybe_save_or_show("roc.png", args.save_plots)

            plt.figure()
            plt.plot(rec, prec)
            plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plot_and_maybe_save_or_show("pr.png", args.save_plots)
    except Exception as e:
        print(f"WARNING: ROC/PR plotting failed: {e}")

    # -------------------------
    # Training curves
    # -------------------------
    if args.plots or args.save_plots:
        h = history.history

        plt.figure()
        plt.plot(h.get("accuracy", []), label="train")
        plt.plot(h.get("val_accuracy", []), label="val")
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plot_and_maybe_save_or_show("accuracy.png", args.save_plots)

        plt.figure()
        plt.plot(h.get("auc", []), label="train")
        plt.plot(h.get("val_auc", []), label="val")
        plt.title("ROC-AUC (metric)")
        plt.xlabel("epoch")
        plt.ylabel("auc")
        plt.legend()
        plot_and_maybe_save_or_show("auc.png", args.save_plots)

        plt.figure()
        plt.plot(h.get("pr_auc", []), label="train")
        plt.plot(h.get("val_pr_auc", []), label="val")
        plt.title("PR-AUC (metric)")
        plt.xlabel("epoch")
        plt.ylabel("pr_auc")
        plt.legend()
        plot_and_maybe_save_or_show("pr_auc.png", args.save_plots)

        plt.figure()
        plt.plot(h.get("loss", []), label="train")
        plt.plot(h.get("val_loss", []), label="val")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plot_and_maybe_save_or_show("loss.png", args.save_plots)


if __name__ == "__main__":
    main()
