"""
Stage 3a — Prepare CNN Training Data
Uses the Kaggle Stock Chart Patterns dataset (CSV + images) and
samples no_pattern images from our generated charts.

Output structure:
  data/cnn_ready/
    train/
      double_top/       (~136 images)
      double_bottom/    (~136 images)
      no_pattern/       (~400 images)
    val/
      double_top/       (~34 images)
      double_bottom/    (~34 images)
      no_pattern/       (~100 images)

Run: python src/prepare_cnn_data.py
"""

import shutil
import random
from pathlib import Path
from collections import Counter
import pandas as pd

random.seed(42)

KAGGLE_DIR  = Path(__file__).parent.parent / "data" / "external" / "kaggle_patterns"
CHARTS_DIR  = Path(__file__).parent.parent / "data" / "charts"
OUT_DIR     = Path(__file__).parent.parent / "data" / "cnn_ready"

TRAIN_SPLIT = 0.70      # 70% train
VAL_SPLIT   = 0.15      # 15% val
TEST_SPLIT  = 0.15      # 15% test
NO_PATTERN_SAMPLES = 500  # number of no_pattern images to sample

# Map Kaggle class names → our folder names
CLASS_MAP = {
    "Double top":    "double_top",
    "Double bottom": "double_bottom",
}


def prepare_kaggle_classes():
    csv_path = KAGGLE_DIR / "Patterns.csv"
    if not csv_path.exists():
        print(f"  ERROR: Patterns.csv not found at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    print(f"\nKaggle dataset: {len(df)} images, {df['ClassName'].nunique()} classes")

    counts = Counter()
    for _, row in df.iterrows():
        cls_name = CLASS_MAP.get(row['ClassName'])
        if cls_name is None:
            continue

        img_path = KAGGLE_DIR / row['Path']
        if not img_path.exists():
            continue

        # Determine train / val / test (70/15/15)
        r = random.random()
        if r < TRAIN_SPLIT:
            split = "train"
        elif r < TRAIN_SPLIT + VAL_SPLIT:
            split = "val"
        else:
            split = "test"

        dest  = OUT_DIR / split / cls_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest / img_path.name)
        counts[f"{split}/{cls_name}"] += 1

    for k, v in sorted(counts.items()):
        print(f"  {k:<30} {v} images")
    return True


def prepare_no_pattern():
    """Sample images from our generated charts as no_pattern class."""
    all_charts = list(CHARTS_DIR.rglob("*.png"))
    if not all_charts:
        print("  WARNING: No generated charts found in data/charts/")
        return False

    # Exclude any that already went to cnn_ready
    random.shuffle(all_charts)
    samples = all_charts[:NO_PATTERN_SAMPLES]

    n_val   = int(len(samples) * VAL_SPLIT)
    n_test  = int(len(samples) * TEST_SPLIT)
    n_train = len(samples) - n_val - n_test

    for i, img_path in enumerate(samples):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        dest = OUT_DIR / split / "no_pattern"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest / img_path.name)

    print(f"\n  no_pattern:")
    print(f"  train/no_pattern              {n_train} images")
    print(f"  val/no_pattern                {n_val} images")
    print(f"  test/no_pattern               {n_test} images")
    return True


def print_summary():
    print(f"\n── Final Dataset Summary ────────────────────────")
    total = 0
    for split in ["train", "val"]:
        split_dir = OUT_DIR / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            count = len(list(cls_dir.glob("*")))
            total += count
            print(f"  {split:<6} / {cls_dir.name:<20} {count} images")
    print(f"  {'TOTAL':<28} {total} images")
    print(f"\n  Output: {OUT_DIR}")
    print("  Ready for: python src/train_cnn.py")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("── Preparing CNN training data ──────────────────")
    kaggle_ok     = prepare_kaggle_classes()
    no_pattern_ok = prepare_no_pattern()

    if kaggle_ok and no_pattern_ok:
        print_summary()
    else:
        print("\nSome steps failed — check errors above.")


if __name__ == "__main__":
    main()
