"""
Stage 4 — Model Evaluation on Held-Out Test Set
Computes final KPI metrics dynamically for whatever class list exists
under <data_dir>/test/.  Works for 3-class (legacy) or N-class setups.

  - Weighted F1 score            (target >= 0.85)
  - Macro F1 score               (shown for fairness under imbalance)
  - Per-class Precision/Recall/F1
  - End-to-end latency           (target <= 3 seconds)
  - Confusion matrix (raw + row-normalised)
  - Error analysis (top confused pairs, calibration)
  - Data-leakage guard: asserts train/val/test filename sets are disjoint

Run:
  python src/evaluate.py
  python src/evaluate.py --data_dir data/cnn_clean --model efficientnet_v2_s
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Device ───────────────────────────────────────────────────────────
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model Builders (must mirror train_cnn.py) ────────────────────────
def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def leakage_guard(data_dir: Path) -> None:
    """Assert that train/val/test filename sets are disjoint."""
    splits = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        splits[split] = {p.name for p in split_dir.rglob("*.png")} | \
                        {p.name for p in split_dir.rglob("*.jpg")}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if a in splits and b in splits:
            overlap = splits[a] & splits[b]
            if overlap:
                raise RuntimeError(
                    f"DATA LEAKAGE: {len(overlap)} file(s) appear in both "
                    f"'{a}' and '{b}' splits of {data_dir}. "
                    f"First 5: {sorted(overlap)[:5]}"
                )
    print(f"  Leakage guard: OK (no filename overlap across splits)")


def run_evaluation(model_path: Path, data_dir: Path, model_name: str,
                    batch_size: int = 32):
    print(f"\n── Evaluating: {model_path.name} on {data_dir.name} ──")
    leakage_guard(data_dir)

    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"  Test directory not found: {test_dir}")
        print("  Make sure <data_dir>/test/ exists with class subfolders.")
        return

    test_ds = ImageFolder(str(test_dir), transform=TEST_TRANSFORM)
    classes = list(test_ds.classes)
    num_classes = len(classes)
    print(f"  Test samples: {len(test_ds)} | Classes ({num_classes}): {classes}")

    # Build and load model
    model = build_model(model_name, num_classes)
    state = torch.load(str(model_path), map_location=DEVICE)
    # Accept both plain state_dict and dicts that wrap it
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    nw = 0 if str(DEVICE) == "mps" else 4
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=nw)

    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = probs.argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
    inference_time    = time.time() - t0
    latency_per_sample_ms = inference_time / max(len(test_ds), 1) * 1000

    # ── KPI Report ────────────────────────────────────────────────────
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_f1    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    per_prec    = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_rec     = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_f1      = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"\n── KPI Results ──────────────────────────────────")
    f1_status  = "✓ PASS" if weighted_f1 >= 0.85 else "✗ FAIL (target: >=0.85)"
    lat_status = "✓ PASS" if latency_per_sample_ms <= 3000 else "✗ FAIL (target: <=3s)"
    print(f"  Weighted F1:          {weighted_f1:.4f}   {f1_status}")
    print(f"  Macro F1:             {macro_f1:.4f}")
    print(f"  Avg latency/sample:   {latency_per_sample_ms:.1f} ms  {lat_status}")
    print(f"\n  Per-class metrics (target Precision >= 0.80):")
    for c, p, r, f in zip(classes, per_prec, per_rec, per_f1):
        ok = "✓" if p >= 0.80 else "✗"
        print(f"    {ok} {c:<28} P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    print(f"\n── Full Classification Report ──")
    print(classification_report(all_labels, all_preds,
                                  target_names=classes, zero_division=0))

    # ── Confusion Matrices (raw + normalised) ─────────────────────────
    cm_raw  = confusion_matrix(all_labels, all_preds)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm_raw, annot=True, fmt='d', ax=axes[0],
                 xticklabels=classes, yticklabels=classes, cmap='Blues')
    axes[0].set_title('Confusion Matrix (raw counts)')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=axes[1],
                 xticklabels=classes, yticklabels=classes, cmap='Blues',
                 vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (row-normalised = recall)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    cm_path = MODELS_DIR / "test_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix: {cm_path}")

    # ── Calibration / Error Analysis ──────────────────────────────────
    probs_arr  = np.array(all_probs)
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    max_probs  = probs_arr.max(axis=1)
    wrong_mask = preds_arr != labels_arr

    # Brier score averaged across classes
    y_true_onehot = np.eye(num_classes)[labels_arr]
    brier = float(np.mean([
        brier_score_loss(y_true_onehot[:, k], probs_arr[:, k])
        for k in range(num_classes)
    ]))

    print(f"\n── Error Analysis ──")
    print(f"  Total errors:                 {wrong_mask.sum()} / {len(all_labels)} "
          f"({wrong_mask.mean()*100:.1f}%)")
    print(f"  Avg confidence on wrong preds: {max_probs[wrong_mask].mean():.3f}" \
          if wrong_mask.any() else "  (no wrong predictions)")
    print(f"  Avg confidence on correct:     {max_probs[~wrong_mask].mean():.3f}" \
          if (~wrong_mask).any() else "")
    print(f"  Brier score (multi-class):     {brier:.4f}   (lower is better-calibrated)")

    confused = {}
    for true, pred in zip(labels_arr[wrong_mask], preds_arr[wrong_mask]):
        pair = (classes[true], classes[pred])
        confused[pair] = confused.get(pair, 0) + 1
    top_confused = sorted(confused.items(), key=lambda x: -x[1])[:5]
    if top_confused:
        print(f"  Top confused pairs:")
        for (t, p), cnt in top_confused:
            print(f"    {t:<28} → {p:<28} ({cnt} times)")

    # ── Save full results CSV ─────────────────────────────────────────
    results_df = pd.DataFrame({
        'true_label': [classes[l] for l in all_labels],
        'pred_label': [classes[p] for p in all_preds],
        'confidence': max_probs,
        'correct':    ~wrong_mask,
    })
    results_path = MODELS_DIR / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Full results saved: {results_path}")

    # ── One-line KPI summary for CI / README ──────────────────────────
    print(
        f"\nSUMMARY  weighted_f1={weighted_f1:.4f} macro_f1={macro_f1:.4f} "
        f"brier={brier:.4f} latency_ms={latency_per_sample_ms:.1f} "
        f"n_classes={num_classes}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cnn_ready_v2",
                        help="Path (relative to project root or absolute) to the "
                             "ImageFolder-style dataset containing train/val/test.")
    parser.add_argument("--model", choices=["efficientnet", "efficientnet_v2_s",
                                              "resnet50", "convnext_tiny"],
                        default="efficientnet")
    parser.add_argument("--weights", type=str, default="models/best_cnn.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    model_path = Path(args.weights)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Run: python src/train_cnn.py")
    else:
        run_evaluation(model_path, data_dir, args.model, args.batch_size)
