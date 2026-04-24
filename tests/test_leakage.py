"""
Data-leakage unit tests.

Run:
    python -m pytest tests/
    # or without pytest:
    python tests/test_leakage.py
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _image_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]


def _by_split(data_dir: Path):
    splits = {}
    for split in ("train", "val", "test"):
        d = data_dir / split
        if d.exists():
            splits[split] = _image_files(d)
    return splits


def test_no_filename_overlap():
    """No image filename should appear in more than one split."""
    for ds_name in ("cnn_ready_v2", "cnn_clean", "cnn_hybrid"):
        data_dir = PROJECT_ROOT / "data" / ds_name
        if not data_dir.exists():
            continue
        splits = _by_split(data_dir)
        names = {s: {p.name for p in files} for s, files in splits.items()}
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            if a in names and b in names:
                overlap = names[a] & names[b]
                assert not overlap, (
                    f"[{ds_name}] {len(overlap)} filename(s) overlap "
                    f"between '{a}' and '{b}'.  First 5: {sorted(overlap)[:5]}"
                )


def test_no_content_duplicates_across_splits():
    """Also guard against same image bytes under different filenames."""
    for ds_name in ("cnn_ready_v2", "cnn_clean", "cnn_hybrid"):
        data_dir = PROJECT_ROOT / "data" / ds_name
        if not data_dir.exists():
            continue
        splits = _by_split(data_dir)
        hashes = {}
        for split, files in splits.items():
            for f in files:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in hashes and hashes[h] != split:
                    raise AssertionError(
                        f"[{ds_name}] identical image bytes in splits "
                        f"'{hashes[h]}' and '{split}'  ({f.name})"
                    )
                hashes[h] = split


def test_classes_nonempty():
    """Every class folder must contain at least one sample per split."""
    for ds_name in ("cnn_ready_v2", "cnn_clean", "cnn_hybrid"):
        data_dir = PROJECT_ROOT / "data" / ds_name
        if not data_dir.exists():
            continue
        for split in ("train", "val", "test"):
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                n = sum(1 for _ in cls_dir.iterdir())
                assert n > 0, f"[{ds_name}/{split}/{cls_dir.name}] is empty"


if __name__ == "__main__":
    failed = 0
    for name, fn in [
        ("filename overlap",   test_no_filename_overlap),
        ("content duplicates", test_no_content_duplicates_across_splits),
        ("classes non-empty",  test_classes_nonempty),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
