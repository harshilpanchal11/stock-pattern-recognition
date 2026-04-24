"""
Stage 2b (NEW) — Download a *multi-class* labelled pattern dataset.

We pull TWO complementary open-source datasets from HuggingFace and
stage them under data/external/ :

  1. foduucom/stockmarket-pattern-detection
     - Dataset (images + YOLO bounding-box labels) that matches the
       YOLOv8 weights already in models/model.pt.
     - 6 classes: Head-and-shoulders-top, Head-and-shoulders-bottom,
                   M_Head, W_Bottom, Triangle, StockLine.
     - Saves to: data/external/foduucom_patterns/

  2. rishi-dua/chart-pattern-recognition-dataset   (optional)
     - Pre-built ImageFolder with 9 pattern classes.
     - Useful for quick multi-class classification training.
     - Saves to: data/external/rishi_patterns/

Both downloads are resumable and idempotent — rerunning skips files
that already exist locally.

Run:
    python src/download_multiclass.py
    python src/download_multiclass.py --skip-rishi
"""

import argparse
from pathlib import Path

EXT_DIR = Path(__file__).parent.parent / "data" / "external"
EXT_DIR.mkdir(parents=True, exist_ok=True)


def download_foduucom_dataset() -> bool:
    target = EXT_DIR / "foduucom_patterns"
    print(f"\n── Downloading foduucom/stockmarket-pattern-detection ──")
    print(f"   Target: {target}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="foduucom/stockmarket-pattern-detection",
            repo_type="dataset",
            local_dir=str(target),
            ignore_patterns=["*.git*", "*.md"],
        )
        n_img = len(list(target.rglob("*.jpg"))) + len(list(target.rglob("*.png")))
        n_lbl = len(list(target.rglob("*.txt")))
        print(f"   ✓ {n_img} images, {n_lbl} YOLO label files")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   If this 404s, the dataset may have been renamed.  Check")
        print("   https://huggingface.co/datasets?search=stockmarket+pattern")
        return False


def download_rishi_dataset() -> bool:
    target = EXT_DIR / "rishi_patterns"
    print(f"\n── Downloading rishi-dua/chart-pattern-recognition-dataset ──")
    print(f"   Target: {target}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="rishi-dua/chart-pattern-recognition-dataset",
            repo_type="dataset",
            local_dir=str(target),
            ignore_patterns=["*.git*", "*.md"],
        )
        n_img = len(list(target.rglob("*.jpg"))) + len(list(target.rglob("*.png")))
        print(f"   ✓ {n_img} images")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   If this 404s, try a Roboflow-Universe alternative — see")
        print("   Group3_Project_Audit_and_Fix_Plan.md §4.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-foduucom", action="store_true")
    parser.add_argument("--skip-rishi",    action="store_true")
    args = parser.parse_args()

    print("Starting multi-class dataset downloads...")
    fod_ok   = True if args.skip_foduucom else download_foduucom_dataset()
    rishi_ok = True if args.skip_rishi    else download_rishi_dataset()

    print("\n── Download Summary ──")
    print(f"  foduucom   : {'✓' if fod_ok   else '✗ FAILED'}")
    print(f"  rishi-dua  : {'✓' if rishi_ok else '✗ FAILED'}")
    if fod_ok:
        print("\nNext step: python src/prepare_multiclass_data.py")
