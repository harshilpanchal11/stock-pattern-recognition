"""
Stage 3a — Prepare CNN Training Data
Converts YOLO-format HuggingFace dataset into ImageFolder structure
required by PyTorch's torchvision.datasets.ImageFolder.

Expected input structure (HuggingFace download):
  data/external/hf_yolov8/
    train/images/*.jpg
    train/labels/*.txt
    valid/images/*.jpg
    valid/labels/*.txt

Output structure:
  data/cnn_ready/
    train/
      head_and_shoulders/
      double_top/
      ...
    val/
      ...

Run: python src/prepare_cnn_data.py
"""

import shutil
from pathlib import Path
from collections import Counter

HF_DIR  = Path(__file__).parent.parent / "data" / "external" / "hf_yolov8"
OUT_DIR = Path(__file__).parent.parent / "data" / "cnn_ready"

CLASSES = {
    0: "head_and_shoulders",
    1: "double_top",
    2: "double_bottom",
    3: "triangle",
    4: "wedge",
    5: "no_pattern",
}


def get_class_from_label_file(lbl_path: Path) -> str:
    """Read first annotation line and return class name."""
    if not lbl_path.exists():
        return "no_pattern"
    with open(lbl_path) as f:
        line = f.readline().strip()
    if not line:
        return "no_pattern"
    cls_id = int(line.split()[0])
    return CLASSES.get(cls_id, "no_pattern")


def prepare_split(img_dir: Path, lbl_dir: Path, split: str):
    imgs = list(img_dir.glob("*.jpg")) + \
           list(img_dir.glob("*.png")) + \
           list(img_dir.glob("*.jpeg"))

    if not imgs:
        print(f"  WARNING: No images found in {img_dir}")
        return Counter()

    print(f"\nProcessing '{split}': {len(imgs)} images...")
    counts = Counter()

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        cls_name = get_class_from_label_file(lbl_path)

        dest = OUT_DIR / split / cls_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest / img_path.name)
        counts[cls_name] += 1

    return counts


def check_imbalance(counts: Counter, split: str):
    """Warn if any class is severely underrepresented."""
    if not counts:
        return
    max_count = max(counts.values())
    print(f"\n  {split} class distribution:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max_count * 20)
        pct = cnt / sum(counts.values()) * 100
        flag = " ⚠ IMBALANCED" if cnt < max_count * 0.2 else ""
        print(f"    {cls:<25} {cnt:>5}  {bar} ({pct:.1f}%){flag}")


def main():
    if not HF_DIR.exists():
        print(f"HuggingFace dataset not found at: {HF_DIR}")
        print("Run src/download_datasets.py first.")
        return

    # Try common directory naming conventions from HuggingFace
    split_map = [
        ("train", HF_DIR / "train" / "images", HF_DIR / "train" / "labels"),
        ("val",   HF_DIR / "valid" / "images", HF_DIR / "valid" / "labels"),
        ("val",   HF_DIR / "val"   / "images", HF_DIR / "val"   / "labels"),
    ]

    seen_splits = set()
    for split, img_dir, lbl_dir in split_map:
        if split in seen_splits:
            continue
        if img_dir.exists():
            counts = prepare_split(img_dir, lbl_dir, split)
            check_imbalance(counts, split)
            seen_splits.add(split)

    print(f"\n── Data preparation complete ───────────────────")
    print(f"  Output: {OUT_DIR}")
    print("  Ready for: python src/train_cnn.py")


if __name__ == "__main__":
    main()
