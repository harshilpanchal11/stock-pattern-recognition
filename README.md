# Stock Chart Pattern Recognition — Group 3
## AI-Driven Pattern Detection using Deep Learning | Spring 2026

Real-time detection of classical chart patterns (Head & Shoulders, Double Top/Bottom,
Triangle, Wedge) using EfficientNet-B0 + YOLOv8 on live candlestick chart images.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run data ingestion (Person A)
```bash
python src/ingest.py
```

### 3. Generate chart images (Person A)
```bash
# Full run (~1-2 hours)
python src/generate_charts.py

# Fast demo mode (~5 min)
python src/generate_charts.py --demo
```

### 4. Download external datasets (Person B)
```bash
python src/download_datasets.py
```

### 5. Prepare CNN training data (Person B)
```bash
python src/prepare_cnn_data.py
```

### 6. Train CNN model (Person B) — ~1.5-2.5 hrs on Apple Silicon
```bash
python src/train_cnn.py
# Or ResNet-50:
python src/train_cnn.py --model resnet50 --epochs 30
```

### 7. Evaluate on test set (Person B)
```bash
python src/evaluate.py
```

### 8. Run Streamlit app locally (Person C)
```bash
streamlit run app/streamlit_app.py
```

### 9. Deploy to Streamlit Community Cloud
- Push to GitHub
- Go to share.streamlit.io
- Connect repo → set main file to `app/streamlit_app.py`
- Deploy!

---

## Project Structure
```
stock-pattern-recognition/
├── data/
│   ├── raw/                  # Parquet files (yfinance output)
│   ├── charts/               # Generated PNG chart images
│   │   ├── train/            # 2020–2023 data
│   │   ├── val/              # 2024 data
│   │   └── test/             # 2025–2026 data
│   └── external/             # HuggingFace + Kaggle datasets
├── src/
│   ├── ingest.py             # Stage 1: yfinance data pull
│   ├── generate_charts.py    # Stage 2: mplfinance chart generation
│   ├── download_datasets.py  # Download HuggingFace + Kaggle data
│   ├── prepare_cnn_data.py   # Convert YOLO dataset → ImageFolder
│   ├── train_cnn.py          # Stage 3: EfficientNet/ResNet training
│   └── evaluate.py           # KPI evaluation on test set
├── models/
│   ├── best_cnn.pth          # Trained CNN weights
│   └── best.pt               # Pre-trained YOLOv8 weights
├── app/
│   └── streamlit_app.py      # Live Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## KPIs
| Metric | Target | Description |
|--------|--------|-------------|
| Weighted F1 | ≥ 0.85 | Across all 6 pattern classes |
| Per-class Precision | ≥ 0.80 | Minimise false positives |
| End-to-end Latency | ≤ 3 sec | Data pull → chart → prediction |

## Pattern Classes
- Head & Shoulders (bearish reversal)
- Double Top (bearish reversal)
- Double Bottom (bullish reversal)
- Triangle (continuation/breakout)
- Wedge (reversal)
- No Pattern

## Tech Stack
`yfinance` · `mplfinance` · `PyTorch` · `EfficientNet-B0` · `YOLOv8` · `Streamlit`

**Total cost: $0**

---

*Group 3 — Images in Finance — Spring 2026*
*⚠️ For educational purposes only. Not financial advice.*
