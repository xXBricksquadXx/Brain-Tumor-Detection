"""
brain_tumor_detector_v2.py

Binary classifier (tumor vs no-tumor) using folder structure:

  brain_tumor_dataset/
    yes/
    no/
    all/   (optional; ignored)

Key features:
- Stratified train/val/test split (val-size is fraction of TRAIN split).
- Pylance-friendly imports.
- Prints confusion matrices + accuracy/sensitivity/specificity at:
    - threshold 0.50
    - chosen threshold (auto-picked or user-specified)
- If --threshold is omitted:
    - threshold-policy=balanced_acc: pick threshold maximizing balanced accuracy on val
    - threshold-policy=min_recall: pick threshold meeting target recall on val, then max specificity
- Saves training curves + ROC + PR curves when --save-plots is used.
- Optional transfer learning (MobileNetV2) + optional two-phase fine-tune:
  Phase 1: frozen backbone (train head)
  Phase 2: unfreeze last N layers (keep BatchNorm frozen) + lower LR

Examples:
  python brain_tumor_detector_v2.py --save-plots

  python brain_tumor_detector_v2.py --threshold 0.60 --save-plots

  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --save-plots

  # hospital-style thresholding (min recall on val, then max specificity):
  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --save-plots \
    --threshold-policy min_recall --target-recall 0.95

  # fine-tune with smoothing + weight decay (recommended if you fine-tune on small data):
  python brain_tumor_detector_v2.py --backbone mobilenetv2 --augment --fine-tune \
    --head-epochs 12 --ft-epochs 10 --ft-lr 5e-5 --unfreeze-last 10 \
    --label-smoothing 0.05 --weight-decay 1e-4 --monitor val_pr_auc --save-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

    p.add_argument("--data-dir", type=str, default="./brain_tumor_dataset",
                   help="Dataset root containing yes/ and no/")

    p.add_argument("--img-h", type=int, default=135, help="Image height")
    p.add_argument("--img-w", type=int, default=240, help="Image width")

    p.add_argument("--test-size", type=float, default=0.33, help="Test split fraction")
    p.add_argument("--val-size", type=float, default=0.30, help="Val fraction of TRAIN split")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Single-stage training (custom CNN or frozen TL if not using fine-tune)
    p.add_argument("--epochs", type=int, default=50, help="Max epochs (single-stage mode)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (single-stage OR head stage)")

    # Two-phase fine-tune options (MobileNetV2 only)
    p.add_argument("--fine-tune", action="store_true",
                   help="Enable two-phase fine-tuning (MobileNetV2 backbone only)")
    p.add_argument("--head-epochs", type=int, default=12,
                   help="Phase 1 epochs (frozen backbone; head training)")
    p.add_argument("--ft-epochs", type=int, default=10,
                   help="Phase 2 epochs (partial unfreeze; fine-tuning)")
    p.add_argument("--ft-lr", type=float, default=5e-5,
                   help="Phase 2 learning rate (fine-tuning)")
    p.add_argument("--unfreeze-last", type=int, default=10,
                   help="How many backbone layers to unfreeze in phase 2 (BatchNorm stays frozen)")

    p.add_argument("--model-out", type=str, default="model.keras", help="Path to save best model")

    p.add_argument("--plots", action="store_true", help="Show plots (GUI)")
    p.add_argument("--save-plots", "--save-plot", dest="save_plots", action="store_true",
                   help="Save plots to PNG files in the repo (accuracy/auc/loss/roc/pr)")

    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold for class=1. If omitted, pick from validation set.")

    # threshold selection policies
    p.add_argument("--threshold-policy", type=str, default="balanced_acc",
                   choices=["balanced_acc", "min_recall"],
                   help="How to auto-pick threshold from validation set when --threshold is omitted.")
    p.add_argument("--target-recall", type=float, default=0.95,
                   help="Used when threshold-policy=min_recall. Target recall on validation set for class=1.")

    p.add_argument("--use-class-weights", action="store_true",
                   help="Apply class weights computed from training labels (often not needed here).")

    p.add_argument("--backbone", type=str, default="custom",
                   choices=["custom", "mobilenetv2"],
                   help="Model backbone: custom (grayscale CNN) or mobilenetv2 (transfer learning).")
    p.add_argument("--backbone-weights", type=str, default="imagenet",
                   choices=["imagenet", "none"],
                   help="Backbone weights. 'imagenet' downloads pretrained weights; 'none' initializes randomly.")
    p.add_argument("--augment", action="store_true",
                   help="Enable lightweight augmentation (recommended for transfer learning).")

    # regularization / calibration helpers
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Binary cross-entropy label smoothing (e.g., 0.05 helps reduce overconfidence).")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (e.g., 1e-4). Uses AdamW if available; else falls back to Adam.")

    # callback monitoring metric
    p.add_argument("--monitor", type=str, default="val_auc",
                   choices=["val_auc", "val_pr_auc"],
                   help="Metric to monitor for early stopping / checkpointing.")

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
        try:
            img = Image.open(p).convert("L")
            img = ImageOps.pad(img, (img_w, img_h), color=0)  # deterministic, preserves aspect
            arr = np.asarray(img, dtype=np.float32) / 255.0   # [0,1]
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

    X = np.array(X_yes + X_no, dtype=np.float32)            # (N,H,W)
    y = np.array(y_yes + y_no, dtype=np.float32)            # (N,)

    X = X[..., None]                                        # (N,H,W,1)
    if channels == 3:
        X = np.repeat(X, 3, axis=-1)                        # (N,H,W,3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_optimizer(lr: float, weight_decay: float) -> keras.optimizers.Optimizer:
    if weight_decay and weight_decay > 0:
        # Prefer AdamW if available (TF/Keras version dependent)
        if hasattr(keras.optimizers, "AdamW"):
            return keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        # Fallback: Adam (no decay)
        return keras.optimizers.Adam(learning_rate=lr)
    return keras.optimizers.Adam(learning_rate=lr)


def compile_binary(model: keras.Model, lr: float, label_smoothing: float, weight_decay: float) -> None:
    loss = keras.losses.BinaryCrossentropy(label_smoothing=float(label_smoothing))
    model.compile(
        optimizer=make_optimizer(lr, weight_decay),
        loss=loss,
        # Note: "accuracy" here is threshold=0.5. Keep it, but rely on AUC/PR-AUC + printed thresholded metrics.
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_model_custom(img_h: int, img_w: int, lr: float, augment: bool, label_smoothing: float, weight_decay: float) -> keras.Model:
    L = keras.layers

    aug = keras.Sequential([
        L.RandomRotation(0.03),
        L.RandomZoom(0.08),
        L.RandomContrast(0.10),
    ], name="augment") if augment else None

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
    compile_binary(model, lr, label_smoothing, weight_decay)
    return model


def build_model_mobilenetv2(
    img_h: int,
    img_w: int,
    lr: float,
    weights: str,
    augment: bool,
    label_smoothing: float,
    weight_decay: float,
) -> Tuple[keras.Model, keras.Model]:
    """
    Returns: (full_model, backbone_model)
    """
    L = keras.layers

    aug = keras.Sequential([
        L.RandomRotation(0.03),
        L.RandomZoom(0.10),
        L.RandomContrast(0.12),
    ], name="augment") if augment else None

    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = inputs

    # X is [0,1] from loader; map to [-1,1] for MobileNetV2
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
    compile_binary(model, lr, label_smoothing, weight_decay)
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


def pick_threshold_balanced_acc(y_true: np.ndarray, probs: np.ndarray) -> float:
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
    Hospital-style: find thresholds where recall >= target_recall, then pick the one
    with the highest specificity (ties broken by higher threshold).
    """
    best_t = None
    best_spec = -1.0

    y_true = y_true.astype(int)

    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        if preds.min() == preds.max():
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        recall = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        if recall >= target_recall:
            if (spec > best_spec) or (spec == best_spec and (best_t is None or t > best_t)):
                best_spec = spec
                best_t = float(t)

    if best_t is not None:
        return best_t

    # If target not reachable, fall back to balanced accuracy
    return pick_threshold_balanced_acc(y_true, probs)


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / max((tn + fp + fn + tp), 1)
    sens = tp / max((tp + fn), 1)  # recall for class 1
    spec = tn / max((tn + fp), 1)  # recall for class 0
    return {"acc": float(acc), "sens": float(sens), "spec": float(spec)}


def print_threshold_block(split_name: str, y_true: np.ndarray, probs: np.ndarray, thr: float) -> None:
    y_true_i = y_true.astype(int)

    for t in [0.50, float(thr)]:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_true_i, preds)
        m = metrics_from_cm(cm)

        tag = f"@{t:.2f}"
        print(f"\n{split_name} Confusion Matrix {tag} (rows=true, cols=pred):")
        print(cm)
        print(f"{split_name} metrics {tag}: acc={m['acc']:.4f} sens(recall1)={m['sens']:.4f} spec(recall0)={m['spec']:.4f}")


def plot_and_maybe_save_or_show(fig_name: str, save_plots: bool) -> None:
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


def make_callbacks(model_out: str, monitor: str) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_out, monitor=monitor, mode="max", save_best_only=True),
    ]


def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data_dir)

    channels = 1 if args.backbone == "custom" else 3
    if args.backbone != "custom":
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
        model = build_model_custom(args.img_h, args.img_w, args.lr, args.augment, args.label_smoothing, args.weight_decay)
    else:
        model, backbone = build_model_mobilenetv2(
            args.img_h, args.img_w, args.lr,
            weights=args.backbone_weights,
            augment=args.augment,
            label_smoothing=args.label_smoothing,
            weight_decay=args.weight_decay,
        )

    model.summary()

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_train)
        print(f"Using class weights: {class_weight}")

    history2 = None

    if args.fine_tune and args.backbone == "mobilenetv2":
        print(f"\n[Phase 1] Frozen backbone head training: epochs={args.head_epochs}, lr={args.lr}")
        history1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.head_epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(args.model_out, args.monitor),
            class_weight=class_weight,
            verbose=1,
        )

        assert backbone is not None
        unfreeze_backbone_for_finetune(backbone, args.unfreeze_last)

        # Re-compile with fine-tune LR (keep smoothing/decay)
        compile_binary(model, args.ft_lr, args.label_smoothing, args.weight_decay)

        print(f"\n[Phase 2] Fine-tune: epochs={args.ft_epochs}, ft_lr={args.ft_lr}, unfreeze_last={args.unfreeze_last}")
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.ft_epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(args.model_out, args.monitor),
            class_weight=class_weight,
            verbose=1,
        )

        history = type("MergedHistory", (), {"history": merge_histories(history1, history2)})()
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(args.model_out, args.monitor),
            class_weight=class_weight,
            verbose=1,
        )

    # Force evaluation on the checkpointed "best" model file for consistency
    best_path = Path(args.model_out)
    if best_path.exists():
        model = keras.models.load_model(best_path)

    test_loss, test_acc, test_auc, test_pr_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}  (threshold=0.50)")
    print(f"Test auc : {test_auc:.4f}")
    print(f"Test pr_auc: {test_pr_auc:.4f}")
    print(f"Saved best model to: {best_path.resolve()}")

    val_probs = model.predict(X_val, verbose=0).reshape(-1)
    print("\nVal prob stats:", f"min={val_probs.min():.3f} mean={val_probs.mean():.3f} max={val_probs.max():.3f}")

    if args.threshold is None:
        if args.threshold_policy == "balanced_acc":
            thr = pick_threshold_balanced_acc(y_val.astype(int), val_probs)
            print(f"Picked threshold from val (balanced_acc): {thr:.2f}")
        else:
            thr = pick_threshold_min_recall(y_val.astype(int), val_probs, target_recall=float(args.target_recall))
            print(f"Picked threshold from val (min_recall target={args.target_recall:.2f}): {thr:.2f}")
    else:
        thr = float(args.threshold)
        print(f"Using user threshold: {thr:.2f}")

    print_threshold_block("Val", y_val, val_probs, thr)

    test_probs = model.predict(X_test, verbose=0).reshape(-1)
    print("\nTest prob stats:", f"min={test_probs.min():.3f} mean={test_probs.mean():.3f} max={test_probs.max():.3f}")

    print_threshold_block("Test", y_test, test_probs, thr)

    test_preds_thr = (test_probs >= float(thr)).astype(int)
    print(f"\nClassification Report @{thr:.2f}:")
    print(classification_report(y_test.astype(int), test_preds_thr, digits=4, zero_division=0))

    # ROC + PR curves on TEST
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

    # Training curves
    if args.plots or args.save_plots:
        h = history.history

        plt.figure()
        plt.plot(h.get("accuracy", []), label="train")
        plt.plot(h.get("val_accuracy", []), label="val")
        plt.title("Accuracy (threshold=0.50)")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plot_and_maybe_save_or_show("accuracy.png", args.save_plots)

        plt.figure()
        plt.plot(h.get("auc", []), label="train")
        plt.plot(h.get("val_auc", []), label="val")
        plt.title("ROC AUC")
        plt.xlabel("epoch")
        plt.ylabel("auc")
        plt.legend()
        plot_and_maybe_save_or_show("auc.png", args.save_plots)

        plt.figure()
        plt.plot(h.get("pr_auc", []), label="train")
        plt.plot(h.get("val_pr_auc", []), label="val")
        plt.title("PR AUC")
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
