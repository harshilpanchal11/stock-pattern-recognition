# Stock Chart Pattern Recognition

Real-time detection of classical chart patterns (Head & Shoulders, Double Top/Bottom,
Triangle, Wedge, Flag, Cup & Handle, …) using a transfer-learnt CNN classifier
(EfficientNet-B0 / EfficientNet-V2-S / ConvNeXt-Tiny) and a YOLOv8 detector on
live candlestick chart images.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest OHLCV data
```bash
python src/ingest.py
```

### 3. Generate chart images
```bash
# Full run (~1-2 hours)
python src/generate_charts.py

# Fast demo (~5 min, test split only)
python src/generate_charts.py --demo
```

### 4A. LEGACY 3-class path (Double-Top / Double-Bottom / No-pattern)
```bash
# Weak-label our own charts
python src/label_charts.py

# Train on the 3-class dataset
python src/train_cnn.py --data_dir data/cnn_ready_v2
python src/evaluate.py  --data_dir data/cnn_ready_v2
```

### 4B. RECOMMENDED multi-class path — HYBRID (7 classes, zero external deps)

We train on a **hybrid** dataset that combines:

1. **Real yfinance charts** (from `data/cnn_ready_v2/`, weak-labeled
   by `src/label_charts.py` into `double_top`, `double_bottom`,
   `no_pattern`) — teaches the CNN real-chart textures.
2. **Synthetic parametric charts** (from `data/cnn_clean/`, rendered
   with the canonical mplfinance style) — provides supervision for
   patterns the weak labeller can't recover (triangles, H&S).

Synthetic-only training transfers poorly to real charts (the model
becomes distribution-bound and predicts `no_pattern` for everything
off-distribution).  The hybrid dataset solves this.

```bash
# 0. (optional) grab YOLOv8 weights for the detector tab
python src/download_datasets.py

# 1. Render real candlestick charts (uses data/raw/*.parquet from ingest.py)
python src/generate_charts.py           # full ~1-2 hours
python src/generate_charts.py --demo    # fast smoke test

# 2. Weak-label the real charts into double_top / double_bottom / no_pattern
python src/label_charts.py

# 3. Generate synthetic 7-class data (~30-45 min)
python src/generate_synthetic_patterns.py

# 4. Merge into the hybrid dataset
python src/build_hybrid_dataset.py

# 5. Leakage guard
python tests/test_leakage.py

# 6. Train (EfficientNet-V2-S, 30 epochs, ~35 min on Apple Silicon)
python src/train_cnn.py --data_dir data/cnn_hybrid --model efficientnet_v2_s --epochs 30

# 7. Evaluate on held-out test split
python src/evaluate.py --data_dir data/cnn_hybrid --model efficientnet_v2_s

# 8. Qualitative real-chart test via the Streamlit dashboard
streamlit run app/streamlit_app.py
```

The hybrid builder is idempotent — rerun whenever synthetic or real
data changes.  Tune `--syn-ratio` to rebalance:

```bash
python src/build_hybrid_dataset.py --syn-ratio 1.0  # strict match to real
python src/build_hybrid_dataset.py --syn-ratio 0    # keep all synthetic
```

If you ever get access to a real multi-class labelled dataset (e.g. via
Roboflow Universe), drop it under `data/external/` and run
`python src/prepare_multiclass_data.py` to merge it in before training.

### 5. Run Streamlit app
```bash
streamlit run app/streamlit_app.py
```
The app reads `models/classes.json` (written by `train_cnn.py`) so the UI
automatically reflects whichever class set you trained on.

### 6. Deploy
Push to GitHub → share.streamlit.io → set main file to `app/streamlit_app.py`.

---

## Data-leakage guard

After running either path, you can sanity-check with:
```bash
python tests/test_leakage.py
```
The test asserts no filename or identical image bytes appear in more than one
split, and that every class folder is non-empty.

---

## Project Structure
```
stock-pattern-recognition/
├── data/
│   ├── raw/                       # Parquet OHLCV files
│   ├── charts/                    # Generated candlestick PNGs
│   ├── external/
│   │   ├── foduucom_patterns/     # YOLO-format multi-class dataset
│   │   ├── rishi_patterns/        # 9-class ImageFolder (optional)
│   │   └── kaggle_patterns/       # legacy 2-class dataset
│   ├── cnn_ready_v2/              # 3-class (real, weak-labelled)
│   ├── cnn_clean/                 # 7-class (synthetic, canonical-style)
│   ├── cnn_hybrid/                # 7-class HYBRID (real + synthetic — RECOMMENDED)
│   └── _deprecated_cnn_ready/     # archived — DO NOT USE (style leak)
├── src/
│   ├── ingest.py                  # Stage 1
│   ├── generate_charts.py         # Stage 2
│   ├── download_datasets.py           # YOLOv8 weights + legacy Kaggle
│   ├── download_multiclass.py         # HF datasets (both currently 404)
│   ├── label_charts.py                # legacy 3-class weak labeller
│   ├── prepare_multiclass_data.py     # YOLO → ImageFolder (if dataset present)
│   ├── generate_synthetic_patterns.py # 7-class synthetic (canonical style)
│   ├── build_hybrid_dataset.py        # (NEW) merges real + synthetic
│   ├── train_cnn.py                   # dynamic-class CNN trainer
│   └── evaluate.py                    # evaluation w/ leakage guard
├── models/
│   ├── best_cnn.pth               # trained CNN weights
│   ├── classes.json               # class list + model name manifest
│   ├── model.pt                   # YOLOv8 weights (foduucom)
│   ├── confusion_matrix_*.png
│   ├── test_confusion_matrix.png
│   └── training_history.csv
├── app/
│   └── streamlit_app.py           # live dashboard
├── tests/
│   └── test_leakage.py
├── requirements.txt               # pinned versions
└── README.md
```

---

## KPIs
| Metric | Target | Description |
|--------|--------|-------------|
| Weighted F1 | ≥ 0.80 | Across all classes |
| Macro F1 | ≥ 0.75 | Fair across imbalance |
| Per-class Precision | ≥ 0.80 | Minimise false positives |
| End-to-end Latency | ≤ 3 sec | Data pull → chart → prediction |

> The legacy README target of weighted-F1 ≥ 0.85 is **achievable only on the
> multi-class path with the foduucom dataset**.  With the 3-class weak-label
> dataset, expect a realistic ceiling around 0.65.

---

## Pattern Classes (synthetic multi-class path — 7 classes)
- Double Top
- Double Bottom
- Head & Shoulders (top)
- Head & Shoulders (bottom / inverse)
- Ascending Triangle
- Descending Triangle
- No Pattern

---

## Tech Stack
`yfinance` · `mplfinance` · `PyTorch` · `EfficientNet-V2-S` · `YOLOv8` · `Streamlit`

**Total cost: $0**

---

*Group 3 — Images in Finance — Spring 2026*
*⚠️educational purposes only. Not financial advice.*
