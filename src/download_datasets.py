"""
Downloads external labeled datasets needed for model training.

1. HuggingFace: foduucom/stockmarket-pattern-detection-yolov8
   - 9,000+ annotated chart images (YOLO format)
   - Saves to: data/external/hf_yolov8/

2. Kaggle: Stock Chart Patterns dataset
   - Pre-labeled images across pattern classes
   - Saves to: data/external/kaggle_patterns/

Run: python src/download_datasets.py
"""

import subprocess
import sys
from pathlib import Path

EXT_DIR = Path(__file__).parent.parent / "data" / "external"
EXT_DIR.mkdir(parents=True, exist_ok=True)


def download_huggingface():
    print("\n── Downloading HuggingFace dataset ─────────────")
    print("  foduucom/stockmarket-pattern-detection-yolov8")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id="foduucom/stockmarket-pattern-detection-yolov8",
            repo_type="dataset",
            local_dir=str(EXT_DIR / "hf_yolov8"),
            ignore_patterns=["*.git*", "*.md"]
        )
        print(f"  ✓ Saved to: {path}")
        return True
    except ImportError:
        print("  Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                                "huggingface_hub", "-q",
                                "--break-system-packages"])
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="foduucom/stockmarket-pattern-detection-yolov8",
            repo_type="dataset",
            local_dir=str(EXT_DIR / "hf_yolov8")
        )
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_yolov8_model():
    print("\n── Downloading pre-trained YOLOv8 model weights ─")
    print("  foduucom/stockmarket-pattern-detection-yolov8")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="foduucom/stockmarket-pattern-detection-yolov8",
            filename="best.pt",
            repo_type="model",
            local_dir=str(models_dir)
        )
        print(f"  ✓ YOLOv8 weights saved to: {path}")
        return True
    except Exception as e:
        print(f"  ERROR downloading YOLOv8 weights: {e}")
        print("  Try manually: https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8")
        return False


def download_kaggle():
    print("\n── Downloading Kaggle dataset ───────────────────")
    print("  Requires Kaggle API key at ~/.kaggle/kaggle.json")
    try:
        import kaggle  # noqa
        out_dir = EXT_DIR / "kaggle_patterns"
        out_dir.mkdir(exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "mahmoudnagytarek/stock-chart-patterns",
            "--unzip", "-p", str(out_dir)
        ], check=True)
        print(f"  ✓ Saved to: {out_dir}")
        return True
    except FileNotFoundError:
        print("  kaggle CLI not found. Install: pip install kaggle")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e}")
        print("  Check your Kaggle API key: ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    print("Starting dataset downloads...")
    hf_ok    = download_huggingface()
    yolo_ok  = download_yolov8_model()
    kgl_ok   = download_kaggle()

    print("\n── Download Summary ────────────────────────────")
    print(f"  HuggingFace dataset : {'✓' if hf_ok   else '✗ FAILED'}")
    print(f"  YOLOv8 weights      : {'✓' if yolo_ok else '✗ FAILED'}")
    print(f"  Kaggle dataset      : {'✓' if kgl_ok  else '✗ FAILED'}")

    if hf_ok:
        print("\nNext step: python src/prepare_cnn_data.py")
