"""
Stage 4 — Model Evaluation on Held-Out Test Set (2025–2026)
Computes final KPI metrics:
  - Weighted F1 score (target >= 0.85)
  - Per-class Precision (target >= 0.80)
  - End-to-end latency (target <= 3 seconds)
  - Confusion matrix
  - Error analysis

Run: python src/evaluate.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    confusion_matrix, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

DATA_DIR   = Path(__file__).parent.parent / "data" / "cnn_ready"
MODELS_DIR = Path(__file__).parent.parent / "models"

CLASSES = [
    'head_and_shoulders', 'double_top', 'double_bottom',
    'triangle', 'wedge', 'no_pattern'
]

# ── Device ───────────────────────────────────────────────────────────
DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def load_model(model_path: Path) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, len(CLASSES))
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def run_evaluation(model_path: Path):
    print(f"\n── Evaluating: {model_path.name} ────────────────────")

    model    = load_model(model_path)
    test_dir = DATA_DIR / "test"

    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        print("Make sure data/cnn_ready/test/ exists with class subfolders.")
        return

    test_ds     = ImageFolder(str(test_dir), transform=TEST_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    all_preds, all_labels = [], []
    all_probs = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = probs.argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
    inference_time = time.time() - t0
    latency_per_sample = inference_time / len(test_ds) * 1000  # ms

    # ── KPI Report ────────────────────────────────────────────────────
    weighted_f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    per_cls_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_cls_rec  = recall_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"\n── KPI Results ──────────────────────────────────")
    f1_status   = "✓ PASS" if weighted_f1 >= 0.85 else "✗ FAIL (target: ≥0.85)"
    lat_status  = "✓ PASS" if latency_per_sample <= 3000 else "✗ FAIL (target: ≤3s)"
    print(f"  Weighted F1:        {weighted_f1:.4f}   {f1_status}")
    print(f"  Avg latency/sample: {latency_per_sample:.1f} ms  {lat_status}")
    print(f"\n  Per-class Precision (target ≥ 0.80):")
    for cls, prec, rec in zip(CLASSES, per_cls_prec, per_cls_rec):
        status = "✓" if prec >= 0.80 else "✗"
        print(f"    {status} {cls:<25} Prec: {prec:.4f}  Rec: {rec:.4f}")

    print(f"\n── Full Classification Report ──────────────────")
    print(classification_report(all_labels, all_preds,
                                  target_names=CLASSES, zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────────────
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Test Set Confusion Matrix')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_path = MODELS_DIR / "test_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix: {cm_path}")

    # ── Error Analysis: lowest confidence correct predictions ─────────
    probs_arr   = np.array(all_probs)
    preds_arr   = np.array(all_preds)
    labels_arr  = np.array(all_labels)
    max_probs   = probs_arr.max(axis=1)

    wrong_mask   = preds_arr != labels_arr
    wrong_probs  = max_probs[wrong_mask]
    wrong_labels = labels_arr[wrong_mask]
    wrong_preds  = preds_arr[wrong_mask]

    print(f"\n── Error Analysis ───────────────────────────────")
    print(f"  Total errors: {wrong_mask.sum()} / {len(all_labels)} "
          f"({wrong_mask.mean()*100:.1f}%)")
    print(f"  Avg confidence on wrong predictions: {wrong_probs.mean():.3f}")

    # Most confused pairs
    confused = {}
    for true, pred in zip(wrong_labels, wrong_preds):
        pair = (CLASSES[true], CLASSES[pred])
        confused[pair] = confused.get(pair, 0) + 1
    top_confused = sorted(confused.items(), key=lambda x: -x[1])[:5]
    print(f"  Top confused pairs:")
    for (true, pred), cnt in top_confused:
        print(f"    {true:<25} → {pred:<25} ({cnt} times)")

    # Save full results CSV
    results_df = pd.DataFrame({
        'true_label':  [CLASSES[l] for l in all_labels],
        'pred_label':  [CLASSES[p] for p in all_preds],
        'confidence':  max_probs,
        'correct':     ~wrong_mask
    })
    results_path = MODELS_DIR / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Full results saved: {results_path}")


if __name__ == "__main__":
    model_path = MODELS_DIR / "best_cnn.pth"
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Run: python src/train_cnn.py")
    else:
        run_evaluation(model_path)
