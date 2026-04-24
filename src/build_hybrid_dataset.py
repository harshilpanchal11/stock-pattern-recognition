"""
Stage 3c (NEW) — Build the HYBRID training dataset (Path C).

Why this exists
---------------
Synthetic-only training produced a model that classified every real
yfinance chart as `no_pattern` at inference time.  Diagnosis: the
parametric templates were out-of-distribution relative to real charts.

The fix is to train on BOTH sources:

  1. `data/cnn_clean/` — synthetic 7-class data (with the patched
     renderer that now matches generate_charts.py style).  This
     provides *geometric* supervision for patterns that have no
     real-chart equivalent in cnn_ready_v2 (triangles, H&S).

  2. `data/cnn_ready_v2/` — real yfinance 60-day windows weak-labeled
     into `double_top`, `double_bottom`, and `no_pattern` by
     `src/label_charts.py`.  This teaches the model real-chart
     textures (gaps, volatility spikes, volume variance, etc.).

The builder merges them into `data/cnn_hybrid/` with:
  - Filename prefixes `syn_` and `real_` so sources are traceable and
    filenames can never collide.
  - Per-class capping so real data gets proportional weight on the
    three overlap classes (default: synthetic cap = 1.5 × real count).
  - Both split assignments preserved (real data uses the date-based
    split from generate_charts.py, synthetic uses the stem-hash split).
  - A leakage guard that asserts no filename appears in more than one
    split *within* cnn_hybrid.

Output
------
  data/cnn_hybrid/{train,val,test}/<class>/<prefix>_<orig>.png
  data/cnn_hybrid/manifest.json
  models/classes.json           (regenerated for the hybrid class set)

Run
---
    # Default: synthetic capped at 1.5× the real per-class count
    python src/build_hybrid_dataset.py

    # Keep all synthetic, let WeightedRandomSampler balance during training
    python src/build_hybrid_dataset.py --syn-ratio 0

    # Strict balance: synthetic = real count for overlap classes
    python src/build_hybrid_dataset.py --syn-ratio 1.0
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "cnn_clean"
REAL_DIR      = PROJECT_ROOT / "data" / "cnn_ready_v2"
OUT_DIR       = PROJECT_ROOT / "data" / "cnn_hybrid"
MODELS_DIR    = PROJECT_ROOT / "models"

SPLITS = ("train", "val", "test")

# Which real-data class name maps to which hybrid class name.  Keeping
# this explicit means label_charts.py can rename its classes without
# silently breaking the hybrid build.
REAL_CLASS_MAP = {
    "double_top":    "double_top",
    "double_bottom": "double_bottom",
    "no_pattern":    "no_pattern",
}

# Classes that MUST end up in the hybrid dataset.  Missing ones will
# trigger a warning (not a hard error) so partial builds still work.
EXPECTED_CLASSES = [
    "ascending_triangle",
    "descending_triangle",
    "double_bottom",
    "double_top",
    "head_and_shoulders_bottom",
    "head_and_shoulders_top",
    "no_pattern",
]


def _images_in(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _copy_with_prefix(src: Path, dest_dir: Path, prefix: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{prefix}{src.name}"
    # shutil.copy2 preserves mtime; cheap for up to ~30k files.
    shutil.copy2(src, dest)
    return dest


def _clean_output():
    """Wipe and recreate OUT_DIR.  On sandboxed / locked filesystems
    where delete is refused, we fall back to best-effort cleanup of
    prefixed files — new copies will overwrite whatever remains."""
    if OUT_DIR.exists():
        try:
            shutil.rmtree(OUT_DIR)
        except (PermissionError, OSError):
            # Best-effort: delete only the files we own (prefixed)
            for p in OUT_DIR.rglob("syn_*"):
                try: p.unlink()
                except Exception: pass
            for p in OUT_DIR.rglob("real_*"):
                try: p.unlink()
                except Exception: pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ingest_real() -> Counter:
    """Copy real weak-labeled images with a 'real_' prefix."""
    counts = Counter()
    if not REAL_DIR.exists():
        print(f"  WARNING: {REAL_DIR} not found.  Run:")
        print("    python src/generate_charts.py")
        print("    python src/label_charts.py")
        return counts
    for split in SPLITS:
        split_dir = REAL_DIR / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            hybrid_cls = REAL_CLASS_MAP.get(cls_dir.name)
            if hybrid_cls is None:
                # Skip any unexpected class (shouldn't happen but be safe)
                continue
            dest_dir = OUT_DIR / split / hybrid_cls
            for img in _images_in(cls_dir):
                _copy_with_prefix(img, dest_dir, prefix="real_")
                counts[(split, hybrid_cls)] += 1
    return counts


def ingest_synthetic(real_counts: Counter, syn_ratio: float) -> Counter:
    """
    Copy synthetic images with a 'syn_' prefix.

    For classes that ALSO have real data, cap synthetic per split at
    `syn_ratio × real_count`.  For synthetic-only classes (triangles,
    H&S), copy everything.
    """
    counts = Counter()
    if not SYNTHETIC_DIR.exists():
        print(f"  WARNING: {SYNTHETIC_DIR} not found.  Run:")
        print("    python src/generate_synthetic_patterns.py")
        return counts

    real_classes = {cls for (_, cls) in real_counts}

    for split in SPLITS:
        split_dir = SYNTHETIC_DIR / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            imgs = _images_in(cls_dir)

            # Cap synthetic if there's also real data for this class.
            if syn_ratio > 0 and cls in real_classes:
                n_real = real_counts.get((split, cls), 0)
                if n_real > 0:
                    cap = int(round(syn_ratio * n_real))
                    if cap < len(imgs):
                        # Deterministic shuffle with fixed seed for reproducibility.
                        rng = random.Random(hash((split, cls, SEED)) & 0xffffffff)
                        imgs = rng.sample(imgs, cap)

            dest_dir = OUT_DIR / split / cls
            for img in imgs:
                _copy_with_prefix(img, dest_dir, prefix="syn_")
                counts[(split, cls)] += 1
    return counts


# ── Leakage + integrity guards ───────────────────────────────────────
def assert_no_leakage():
    """No filename should appear in more than one split."""
    seen: dict[str, str] = {}
    for split in SPLITS:
        for p in (OUT_DIR / split).rglob("*.png"):
            prev = seen.get(p.name)
            if prev is not None and prev != split:
                raise AssertionError(
                    f"Leakage: {p.name} appears in both '{prev}' and '{split}'"
                )
            seen[p.name] = split
    print("  Leakage guard: OK (no filename appears in more than one split)")


def assert_classes_nonempty(counts: Counter):
    """Every class × split intersection should have at least 1 sample."""
    for split in SPLITS:
        split_classes = {c for (s, c), n in counts.items() if s == split and n > 0}
        missing = set(EXPECTED_CLASSES) - split_classes
        if missing:
            print(f"  WARNING [{split}]: missing classes {sorted(missing)}")


# ── Reporting ────────────────────────────────────────────────────────
def print_summary(real_counts: Counter, syn_counts: Counter):
    print("\n── Hybrid dataset summary ──")
    all_keys = set(real_counts) | set(syn_counts)
    by_split: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for split, cls in sorted(all_keys):
        r = real_counts.get((split, cls), 0)
        s = syn_counts.get((split, cls), 0)
        by_split[split].append((cls, r, s))

    for split in SPLITS:
        if split not in by_split:
            continue
        total_r = sum(r for _, r, _ in by_split[split])
        total_s = sum(s for _, _, s in by_split[split])
        total   = total_r + total_s
        print(f"\n  {split}:  {total} images  "
              f"({total_r} real  +  {total_s} synthetic)")
        print(f"    {'class':<32} {'real':>6} {'syn':>6} {'total':>6}")
        for cls, r, s in sorted(by_split[split]):
            tag = "" if r > 0 else "  (synthetic-only)"
            print(f"    {cls:<32} {r:>6} {s:>6} {r+s:>6}{tag}")


def write_manifests(real_counts: Counter, syn_counts: Counter):
    all_classes = sorted({c for (_, c) in set(real_counts) | set(syn_counts)})
    # cnn_hybrid/manifest.json
    per_split = defaultdict(dict)
    for split, cls in set(real_counts) | set(syn_counts):
        per_split[split][cls] = {
            "real": real_counts.get((split, cls), 0),
            "syn":  syn_counts.get((split, cls), 0),
        }
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "classes": all_classes,
        "seed":    SEED,
        "source":  "hybrid (real cnn_ready_v2 + synthetic cnn_clean)",
        "splits":  dict(per_split),
    }, indent=2))

    # models/classes.json — regenerate for the hybrid class set so
    # Streamlit and evaluate.py stay in sync.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "classes.json").write_text(json.dumps({
        "classes": all_classes,
        "model":   "efficientnet_v2_s",
        "source":  "hybrid",
        "seed":    SEED,
        "windows": [60],
    }, indent=2))
    print(f"\n  Manifest:      {OUT_DIR / 'manifest.json'}")
    print(f"  classes.json:  {MODELS_DIR / 'classes.json'}")


def main(syn_ratio: float):
    print("── Building hybrid dataset ──")
    print(f"   synthetic source: {SYNTHETIC_DIR}")
    print(f"   real source:      {REAL_DIR}")
    print(f"   output:           {OUT_DIR}")
    print(f"   syn_ratio:        "
          f"{'keep all synthetic' if syn_ratio == 0 else f'{syn_ratio} × real count'}")

    _clean_output()

    real_counts = ingest_real()
    syn_counts  = ingest_synthetic(real_counts, syn_ratio)

    assert_no_leakage()
    assert_classes_nonempty(real_counts + syn_counts)
    print_summary(real_counts, syn_counts)
    write_manifests(real_counts, syn_counts)

    print("\nNext:")
    print("  python tests/test_leakage.py")
    print("  python src/train_cnn.py --data_dir data/cnn_hybrid "
          "--model efficientnet_v2_s --epochs 30")
    print("  python src/evaluate.py --data_dir data/cnn_hybrid "
          "--model efficientnet_v2_s")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--syn-ratio", type=float, default=1.5,
        help="Cap synthetic-per-class at RATIO × real-per-class for "
             "classes with real data.  0 = keep all synthetic.  "
             "Default: 1.5",
    )
    args = parser.parse_args()
    main(args.syn_ratio)
