"""
Stage 3b — CNN Model Training
Transfer-learning fine-tune of EfficientNet-B0 / EfficientNet-V2-S /
ResNet-50 / ConvNeXt-Tiny on any ImageFolder-style dataset.

Key properties:
  - Number of classes is inferred dynamically from the dataset.
  - Fully reproducible (seeds torch, numpy, random, cudnn-deterministic).
  - Label-smoothing CE + class-balanced WeightedRandomSampler.
  - Stronger train-time augmentation to counter label noise.
  - Saves a manifest (classes.json) next to the weights so inference
    scripts and Streamlit can load the class list automatically.

Run:
  python src/train_cnn.py
  python src/train_cnn.py --data_dir data/cnn_clean --model efficientnet_v2_s --epochs 30
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    classification_report, f1_score, precision_score, confusion_matrix,
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
MODELS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Training on: {dev}")
    return dev


# ── Transforms ───────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform() -> transforms.Compose:
    """
    Candlestick-chart-friendly augmentation.
    We deliberately DO NOT horizontally flip:  flipping a Double-Top
    produces a Double-Bottom, which would corrupt labels.  Same for H&S.
    """
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.RandAugment(num_ops=1, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])


VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model Builders ───────────────────────────────────────────────────
def build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ('features.7', 'features.8', 'classifier'))
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model


def build_efficientnet_v2_s(num_classes):
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ('features.6', 'features.7', 'classifier'))
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model


def build_resnet50(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V2')
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ('layer4', 'fc'))
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model


def build_convnext_tiny(num_classes):
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ('features.7', 'classifier'))
    in_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_f, num_classes)
    return model


MODEL_BUILDERS = {
    "efficientnet":       build_efficientnet_b0,
    "efficientnet_v2_s":  build_efficientnet_v2_s,
    "resnet50":           build_resnet50,
    "convnext_tiny":      build_convnext_tiny,
}


# ── Training Loop ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += len(labels)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def save_confusion_matrix(labels, preds, classes, model_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                 xticklabels=classes, yticklabels=classes, cmap='Blues')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    out = MODELS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Confusion matrix saved: {out}")


def main(model_name: str, epochs: int, batch_size: int, lr: float,
         data_dir: str, label_smoothing: float, use_sampler: bool):

    device    = get_device()
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not (data_path / "train").exists():
        print(f"Training data not found at {data_path}")
        return

    train_ds = ImageFolder(str(data_path / "train"), transform=build_train_transform())
    val_ds   = ImageFolder(str(data_path / "val"),   transform=VAL_TRANSFORM)

    classes     = train_ds.classes
    num_classes = len(classes)
    print(f"\nDataset: {len(train_ds)} train | {len(val_ds)} val")
    print(f"Classes ({num_classes}): {classes}")

    # Class-balanced sampling (fixes imbalance without hurting calibration)
    class_counts = np.array([
        len(list((data_path / "train" / c).glob("*"))) for c in classes
    ], dtype=float)
    print(f"Train class counts: { dict(zip(classes, class_counts.astype(int))) }")

    sampler = None
    if use_sampler:
        per_class_weight = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
        sample_weights   = np.array([per_class_weight[y] for _, y in train_ds.samples])
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights), replacement=True,
        )

    nw = 0 if str(device) == "mps" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=(sampler is None), sampler=sampler,
                               num_workers=nw)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                               shuffle=False, num_workers=nw)

    # Model
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {model_name}; choose from {list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[model_name](num_classes).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Train
    best_f1    = 0.0
    history    = []
    model_path = MODELS_DIR / "best_cnn.pth"

    print(f"\n── Training {model_name} for {epochs} epochs ──")
    t0 = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_labels, val_preds = evaluate_loader(model, val_loader, device)
        val_f1   = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_prec = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'val_f1':     val_f1,
            'val_macro_f1': val_macro,
            'val_precision': val_prec,
        })

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            # Write class manifest alongside weights
            with open(MODELS_DIR / "classes.json", "w") as f:
                json.dump({"classes": classes, "model": model_name,
                            "num_classes": num_classes}, f, indent=2)
            marker = "  ← best"

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
              f"Val F1: {val_f1:.4f} | Macro F1: {val_macro:.4f} | "
              f"Prec: {val_prec:.4f}{marker}")

    elapsed = time.time() - t0
    print(f"\n── Training complete in {elapsed/60:.1f} min ──")
    print(f"  Best weighted Val F1: {best_f1:.4f}")
    print(f"  Weights: {model_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(MODELS_DIR / "training_history.csv", index=False)

    # Final report on validation set
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    val_labels, val_preds = evaluate_loader(model, val_loader, device)
    print("\nFinal Classification Report (Val Set):")
    print(classification_report(val_labels, val_preds,
                                  target_names=classes, zero_division=0))
    save_confusion_matrix(val_labels, val_preds, classes, model_name)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(history_df['epoch'], history_df['val_f1'],       label='Val Weighted F1', color='green')
    axes[1].plot(history_df['epoch'], history_df['val_macro_f1'], label='Val Macro F1',    color='purple')
    axes[1].plot(history_df['epoch'], history_df['val_precision'], label='Val Precision',  color='orange')
    axes[1].set_title('Validation Metrics'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "training_curves.png", dpi=100)
    plt.close(fig)
    print(f"  Training curves saved: {MODELS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_BUILDERS.keys()),
                        default="efficientnet")
    parser.add_argument("--epochs",          type=int,   default=25)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--data_dir",        type=str,   default="data/cnn_ready_v2")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_sampler",      action="store_true",
                        help="Disable WeightedRandomSampler (use shuffle=True instead)")
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size, args.lr,
         args.data_dir, args.label_smoothing, not args.no_sampler)
