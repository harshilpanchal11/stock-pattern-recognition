"""
Stage 3b — CNN Model Training
Fine-tunes EfficientNet-B0 (primary) and ResNet-50 (comparison)
via transfer learning on the prepared chart image dataset.

Supports Apple Silicon MPS, NVIDIA CUDA, and CPU.

Run: python src/train_cnn.py
     python src/train_cnn.py --model resnet50 --epochs 30
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (classification_report, f1_score,
                              precision_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data" / "cnn_ready"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASSES = [
    'head_and_shoulders', 'double_top', 'double_bottom',
    'triangle', 'wedge', 'no_pattern'
]
NUM_CLASSES = len(CLASSES)

# ── Device ───────────────────────────────────────────────────────────
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
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Model Builders ───────────────────────────────────────────────────
def build_efficientnet(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # Freeze all but last 2 feature blocks + classifier
    for name, param in model.named_parameters():
        param.requires_grad = any(
            k in name for k in ('features.7', 'features.8', 'classifier')
        )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights='IMAGENET1K_V2')
    # Freeze all but layer4 + fc
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ('layer4', 'fc'))
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

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
        total += len(labels)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds

# ── Confusion Matrix Plot ─────────────────────────────────────────────
def save_confusion_matrix(labels, preds, model_name: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    out = MODELS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Confusion matrix saved: {out}")

# ── Main ─────────────────────────────────────────────────────────────
def main(model_name: str = "efficientnet", epochs: int = 20,
         batch_size: int = 32, lr: float = 3e-4):

    device = get_device()

    # Data
    if not (DATA_DIR / "train").exists():
        print(f"Training data not found at {DATA_DIR}")
        print("Run: python src/prepare_cnn_data.py")
        return

    train_ds = ImageFolder(str(DATA_DIR / "train"), transform=TRAIN_TRANSFORM)
    val_ds   = ImageFolder(str(DATA_DIR / "val"),   transform=VAL_TRANSFORM)

    # num_workers=0 for MPS stability; increase for CUDA/CPU
    nw = 0 if str(device) == "mps" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=nw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=nw)

    print(f"\nDataset: {len(train_ds)} train | {len(val_ds)} val")
    print(f"Classes: {train_ds.classes}\n")

    # Model
    if model_name == "efficientnet":
        model = build_efficientnet(NUM_CLASSES)
    else:
        model = build_resnet50(NUM_CLASSES)
    model = model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_f1    = 0.0
    history    = []
    model_path = MODELS_DIR / "best_cnn.pth"

    print(f"── Training {model_name} for {epochs} epochs ─────────────")
    t0 = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_labels, val_preds = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds,
                           average='weighted', zero_division=0)
        val_prec = precision_score(val_labels, val_preds,
                                    average='weighted', zero_division=0)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_f1': val_f1,
            'val_precision': val_prec
        })

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            marker = "  ← best"

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Acc: {train_acc:.3f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Prec: {val_prec:.4f}{marker}")

    elapsed = time.time() - t0
    print(f"\n── Training complete in {elapsed/60:.1f} min ─────────────")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"  Weights: {model_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(MODELS_DIR / "training_history.csv", index=False)

    # Final report on validation set
    model.load_state_dict(torch.load(model_path, map_location=device))
    val_labels, val_preds = evaluate(model, val_loader, device)
    print("\nFinal Classification Report (Val Set):")
    print(classification_report(val_labels, val_preds,
                                  target_names=CLASSES, zero_division=0))
    save_confusion_matrix(val_labels, val_preds, model_name)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[1].plot(history_df['epoch'], history_df['val_f1'], label='Val F1',
                  color='green')
    axes[1].plot(history_df['epoch'], history_df['val_precision'],
                  label='Val Precision', color='orange')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "training_curves.png", dpi=100)
    plt.close(fig)
    print(f"  Training curves saved: {MODELS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["efficientnet", "resnet50"],
                        default="efficientnet")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size, args.lr)
