"""
Stage 3a (NEW) — Prepare *multi-class* CNN data.

Two input paths are supported:

  A) foduucom YOLO-format dataset (preferred — matches YOLOv8 weights)
     - For every annotated image we crop each bounding box and save the
       crop under data/cnn_clean/<split>/<class>/<file>.jpg
     - Images with no labels become a *controlled* no_pattern sample
       (capped at the majority-class count to avoid dominating the set).

  B) rishi-dua pre-built ImageFolder (optional)
     - We merge it in *as a second source* of labelled images.  Class
       names are normalised (lowercase, underscore) and mapped through
     - CLASS_ALIAS below so foduucom and rishi naming agree.

Splitting is performed by *image stem* (70/15/15) using a hash so
adjacent sliding-window crops never leak across splits.  Seeds are
fixed for reproducibility.

Output: data/cnn_clean/{train,val,test}/<class>/*.jpg

Run:
    python src/prepare_multiclass_data.py
    python src/prepare_multiclass_data.py --sources foduucom
    python src/prepare_multiclass_data.py --no-pattern-from-charts data/charts
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
EXT_DIR      = PROJECT_ROOT / "data" / "external"
CHART_DIR    = PROJECT_ROOT / "data" / "charts"
OUT_DIR      = PROJECT_ROOT / "data" / "cnn_clean"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test takes the remainder (0.15)

# Normalise class labels across the two datasets.  Keys are lowercased
# with underscores.  Unknown classes are dropped with a warning.
CLASS_ALIAS = {
    # foduucom names
    "head and shoulders top":    "head_and_shoulders_top",
    "head and shoulders bottom": "head_and_shoulders_bottom",
    "m_head":                    "m_head",
    "w_bottom":                  "w_bottom",
    "triangle":                  "triangle",
    "stockline":                 "stockline",

    # rishi-dua names (common variants)
    "head_and_shoulders":          "head_and_shoulders_top",
    "inverse_head_and_shoulders":  "head_and_shoulders_bottom",
    "double_top":                  "double_top",
    "double_bottom":               "double_bottom",
    "triple_top":                  "triple_top",
    "triple_bottom":               "triple_bottom",
    "rising_wedge":                "rising_wedge",
    "falling_wedge":               "falling_wedge",
    "flag":                        "flag",
    "cup_and_handle":              "cup_and_handle",
}


def split_from_hash(name: str) -> str:
    """Deterministic 70/15/15 split by filename stem hash."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) / (1 << 128)
    if h < TRAIN_FRAC:                 return "train"
    if h < TRAIN_FRAC + VAL_FRAC:      return "val"
    return "test"


def normalise_class(raw: str) -> str | None:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIAS.get(key, CLASS_ALIAS.get(raw.strip().lower()))


# ── Source A : foduucom YOLO dataset ─────────────────────────────────
def _load_yolo_classes(dataset_root: Path) -> list[str]:
    """Try to find class names from data.yaml or dataset.yaml."""
    for candidate in ("data.yaml", "dataset.yaml", "classes.txt"):
        p = dataset_root / candidate
        if p.exists():
            if p.suffix in {".yaml", ".yml"}:
                # Minimal parser — avoid yaml dep; look for `names:` list.
                text = p.read_text()
                if "names:" in text:
                    after = text.split("names:", 1)[1]
                    # Pull out bracketed list [a, b, c] or dash list
                    if "[" in after and "]" in after:
                        body = after[after.find("[") + 1:after.find("]")]
                        return [x.strip().strip("'\"") for x in body.split(",") if x.strip()]
                    names = []
                    for line in after.splitlines():
                        line = line.strip()
                        if line.startswith("-"):
                            names.append(line.lstrip("- ").strip().strip("'\""))
                        elif line and not line.startswith("#"):
                            break
                    if names:
                        return names
            else:
                return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # Fallback — matches the 6-class foduucom default order.
    return [
        "Head and shoulders top",
        "Head and shoulders bottom",
        "M_Head",
        "W_Bottom",
        "Triangle",
        "StockLine",
    ]


def _iter_foduucom_samples(dataset_root: Path):
    """Yield (image_path, [(class_idx, bbox_xywh_rel)...]) pairs."""
    # YOLO datasets commonly have images/ and labels/ at top level, or
    # inside train/valid/test subfolders.
    for img_path in dataset_root.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Match label by replacing 'images' with 'labels' in the path.
        label_path = None
        for part_swap in ("images", "image"):
            if part_swap in img_path.parts:
                idx = img_path.parts.index(part_swap)
                parts = list(img_path.parts)
                parts[idx] = "labels"
                candidate = Path(*parts).with_suffix(".txt")
                if candidate.exists():
                    label_path = candidate
                    break
        if label_path is None:
            # sibling labels dir
            candidate = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
            if candidate.exists():
                label_path = candidate
        boxes = []
        if label_path and label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_i = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append((cls_i, (x, y, w, h)))
        yield img_path, boxes


def ingest_foduucom(pad: float = 0.05):
    root = EXT_DIR / "foduucom_patterns"
    if not root.exists():
        print(f"  foduucom dataset not found at {root}")
        print("  Run: python src/download_multiclass.py")
        return Counter()

    class_names = _load_yolo_classes(root)
    print(f"  foduucom class list: {class_names}")

    counts = Counter()
    for img_path, boxes in _iter_foduucom_samples(root):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        split = split_from_hash(img_path.stem)

        if not boxes:
            # Unannotated window → no_pattern (capped later)
            dest_dir = OUT_DIR / split / "no_pattern_candidates"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{img_path.stem}.jpg"
            img.save(dest, "JPEG", quality=90)
            counts[(split, "no_pattern_candidates")] += 1
            continue

        for i, (cls_i, (x, y, w, h)) in enumerate(boxes):
            if cls_i >= len(class_names):
                continue
            raw_name  = class_names[cls_i]
            norm_name = normalise_class(raw_name)
            if norm_name is None:
                continue

            # YOLO xywh are fractions of W, H — convert to pixel box
            cx, cy = x * W, y * H
            bw, bh = w * W, h * H
            x0 = max(0, int(cx - bw * (0.5 + pad)))
            y0 = max(0, int(cy - bh * (0.5 + pad)))
            x1 = min(W, int(cx + bw * (0.5 + pad)))
            y1 = min(H, int(cy + bh * (0.5 + pad)))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop = img.crop((x0, y0, x1, y1))

            dest_dir = OUT_DIR / split / norm_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{img_path.stem}_{i}.jpg"
            crop.save(dest, "JPEG", quality=90)
            counts[(split, norm_name)] += 1

    return counts


# ── Source B : rishi-dua pre-built ImageFolder ───────────────────────
def ingest_rishi():
    root = EXT_DIR / "rishi_patterns"
    if not root.exists():
        print(f"  rishi dataset not found at {root} (optional — skipping)")
        return Counter()

    counts = Counter()
    for cls_dir in root.rglob("*"):
        if not cls_dir.is_dir():
            continue
        # Heuristic: only treat leaf dirs that contain images as classes.
        imgs = [p for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            continue
        norm = normalise_class(cls_dir.name)
        if norm is None:
            continue
        for img_path in imgs:
            split = split_from_hash(img_path.stem)
            dest_dir = OUT_DIR / split / norm
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(img_path, dest_dir / img_path.name)
                counts[(split, norm)] += 1
            except Exception:
                continue
    return counts


# ── Cap no_pattern to avoid domination ───────────────────────────────
def finalise_no_pattern(counts: Counter):
    for split in ("train", "val", "test"):
        cand_dir = OUT_DIR / split / "no_pattern_candidates"
        if not cand_dir.exists():
            continue
        # Cap at the majority of labelled classes in this split
        split_counts = [v for (s, c), v in counts.items()
                         if s == split and c != "no_pattern_candidates"]
        if not split_counts:
            cap = 500
        else:
            cap = int(np.median(split_counts))
        imgs = sorted(cand_dir.iterdir())
        random.shuffle(imgs)
        keep = imgs[:cap]
        discard = imgs[cap:]
        dest_dir = OUT_DIR / split / "no_pattern"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in keep:
            img.rename(dest_dir / img.name)
        for img in discard:
            img.unlink(missing_ok=True)
        cand_dir.rmdir()
        counts[(split, "no_pattern")] = len(keep)
        counts.pop((split, "no_pattern_candidates"), None)


def print_summary(counts: Counter):
    print(f"\n── Final dataset summary ──")
    by_split = defaultdict(dict)
    for (split, cls), n in counts.items():
        by_split[split][cls] = n
    for split in ("train", "val", "test"):
        if split not in by_split:
            continue
        total = sum(by_split[split].values())
        print(f"  {split}:  {total} images  ({len(by_split[split])} classes)")
        for cls, n in sorted(by_split[split].items()):
            print(f"     {cls:<30} {n}")
    # Write manifest for training/streamlit
    classes = sorted({c for (_, c), n in counts.items() if n > 0})
    (OUT_DIR / "manifest.json").write_text(json.dumps(
        {"classes": classes, "seed": SEED,
         "splits": {s: dict(by_split[s]) for s in by_split}},
        indent=2,
    ))
    print(f"\n  Manifest: {OUT_DIR / 'manifest.json'}")
    print("  Next: python src/train_cnn.py --data_dir data/cnn_clean --model efficientnet_v2_s")


def main(sources: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("── Preparing multi-class CNN data ──")
    counts = Counter()
    if "foduucom" in sources:
        counts.update(ingest_foduucom())
    if "rishi" in sources:
        counts.update(ingest_rishi())
    finalise_no_pattern(counts)
    print_summary(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        choices=["foduucom", "rishi"],
                        default=["foduucom", "rishi"])
    args = parser.parse_args()
    main(args.sources)
