"""
Group 3 — AI Stock Chart Pattern Recognition
Live Streamlit Dashboard

  - yfinance live OHLCV
  - mplfinance candlestick (display + model-style 224x224)
  - CNN pattern classifier (class list read from models/classes.json)
  - YOLOv8 bounding-box detector
  - End-to-end latency KPI

Run locally : streamlit run app/streamlit_app.py
Deploy      : Streamlit Community Cloud -> share.streamlit.io
"""

import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
CLASSES_JSON = MODELS_DIR / "classes.json"

# Fallback class list if classes.json isn't present (legacy 3-class).
LEGACY_CLASSES = ['double_bottom', 'double_top', 'no_pattern']

# Per-class colours / descriptions.  Keys are matched CASE-INSENSITIVELY
# against the class list loaded at runtime.  Classes not in this map are
# rendered with a default grey.
PATTERN_META = {
    'double_top':               ('#e67e22', 'Bearish reversal — two peaks at a similar level with a trough between.'),
    'double_bottom':            ('#27ae60', 'Bullish reversal — two troughs at a similar level with a peak between.'),
    'no_pattern':               ('#7f8c8d', 'No classical pattern detected with sufficient confidence.'),
    'head_and_shoulders_top':   ('#c0392b', 'Bearish reversal — three peaks, the middle one highest.'),
    'head_and_shoulders_bottom':('#2ecc71', 'Bullish reversal — inverse head & shoulders.'),
    'm_head':                   ('#d35400', 'Bearish — triple-peak M formation.'),
    'w_bottom':                 ('#16a085', 'Bullish — double-trough W formation.'),
    'triangle':                 ('#8e44ad', 'Breakout pattern — converging trendlines.'),
    'wedge':                    ('#2980b9', 'Reversal/continuation — sloping converging trendlines.'),
    'flag':                     ('#1abc9c', 'Brief pause against the main trend; continuation signal.'),
    'cup_and_handle':           ('#f39c12', 'Bullish continuation — rounded bottom plus small consolidation.'),
    'stockline':                ('#34495e', 'Dominant trendline detected by YOLOv8.'),
}

POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL',
    'META', 'AMZN', 'JPM',  'V',    'XOM',
]


def load_class_list():
    """Return (classes, model_name). Falls back to legacy 3-class list."""
    if CLASSES_JSON.exists():
        meta = json.loads(CLASSES_JSON.read_text())
        return meta.get("classes", LEGACY_CLASSES), meta.get("model", "efficientnet")
    return LEGACY_CLASSES, "efficientnet"


def norm_key(cls_name: str) -> str:
    return cls_name.lower().replace(' ', '_').replace('-', '_')


def meta_for(cls_name: str):
    colour, desc = PATTERN_META.get(norm_key(cls_name),
                                     ('#7f8c8d', cls_name))
    return colour, desc


CLASSES, MODEL_NAME = load_class_list()


# ── Model Loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model...")
def load_cnn():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        num_classes = len(CLASSES)
        if MODEL_NAME == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=None)
            in_f  = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
        elif MODEL_NAME == "resnet50":
            model = models.resnet50(weights=None)
            in_f  = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
        elif MODEL_NAME == "convnext_tiny":
            model = models.convnext_tiny(weights=None)
            in_f  = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_f, num_classes)
        else:  # efficientnet b0 (default / legacy)
            model = models.efficientnet_b0(weights=None)
            in_f  = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))

        weights_path = MODELS_DIR / "best_cnn.pth"
        model.load_state_dict(torch.load(str(weights_path), map_location='cpu'))
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, "models/best_cnn.pth not found. Run src/train_cnn.py first."
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo():
    try:
        from ultralytics import YOLO
        weights_path = MODELS_DIR / "model.pt"
        return YOLO(str(weights_path)), None
    except FileNotFoundError:
        return None, "models/model.pt not found. Run src/download_datasets.py first."
    except Exception as e:
        return None, str(e)


def cnn_inference(model, img_pil):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(img_pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return pred_idx, probs


# ── Chart Generation ─────────────────────────────────────────────────
def generate_chart(ohlcv, ticker, show_ma, show_volume):
    add_plots = []
    if show_ma:
        if len(ohlcv) >= 20:
            ma20 = ohlcv['Close'].rolling(20).mean()
            add_plots.append(mpf.make_addplot(ma20, color='#3498db', width=1.2, label='MA20'))
        if len(ohlcv) >= 50:
            ma50 = ohlcv['Close'].rolling(50).mean()
            add_plots.append(mpf.make_addplot(ma50, color='#e67e22', width=1.2, label='MA50'))

    style = mpf.make_mpf_style(
        base_mpf_style='charles', gridstyle='--', gridcolor='#ecf0f1',
        rc={'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7},
    )

    fig, _ = mpf.plot(
        ohlcv, type='candle', style=style,
        addplot=add_plots if add_plots else [],
        volume=show_volume, returnfig=True,
        figsize=(8, 5), tight_layout=True,
        title=f'\n{ticker} — {len(ohlcv)}-Day Candlestick',
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).convert('RGB')


def generate_model_chart(ohlcv):
    """
    Render an inference chart that matches the training pipeline as
    closely as possible.  We deliberately DROP bbox_inches='tight' here
    so the final pixel size is deterministic, then resize to 224x224.
    """
    model_style = mpf.make_mpf_style(
        base_mpf_style='charles', gridstyle='--', gridcolor='#ecf0f1',
        rc={
            'figure.figsize': (2.24, 2.24), 'figure.dpi': 100,
            'axes.labelsize': 0, 'xtick.labelsize': 0, 'ytick.labelsize': 0,
            'axes.spines.top': False, 'axes.spines.right': False,
        },
    )
    add_plots = []
    if len(ohlcv) >= 20:
        ma20 = ohlcv['Close'].rolling(20).mean()
        add_plots.append(mpf.make_addplot(ma20, color='#3498db', width=0.8))
    if len(ohlcv) >= 50:
        ma50 = ohlcv['Close'].rolling(50).mean()
        add_plots.append(mpf.make_addplot(ma50, color='#e67e22', width=0.8))

    fig, _ = mpf.plot(
        ohlcv, type='candle', style=model_style,
        addplot=add_plots if add_plots else [],
        volume=False, returnfig=True, tight_layout=True,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf).convert('RGB').resize((224, 224), Image.LANCZOS)
    return img


# ── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Pattern Detector | Group 3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 13px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Pattern Detector")
    st.markdown("*Group 3 — Images in Finance*")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    ticker = st.text_input(
        "Stock Ticker Symbol", value="AAPL",
        help="Enter any valid ticker (e.g. AAPL, TSLA, NVDA)",
    ).upper().strip()

    st.markdown("**Quick pick:**")
    cols = st.columns(5)
    for i, t in enumerate(POPULAR_TICKERS[:5]):
        if cols[i].button(t, key=f"btn_{t}", use_container_width=True):
            ticker = t
    cols2 = st.columns(5)
    for i, t in enumerate(POPULAR_TICKERS[5:]):
        if cols2[i].button(t, key=f"btn2_{t}", use_container_width=True):
            ticker = t

    st.divider()

    # Window guard: only expose windows the current model was trained on.
    # For the legacy 3-class model we trained on 30-day charts only.
    manifest_windows = [30]
    if CLASSES_JSON.exists():
        meta = json.loads(CLASSES_JSON.read_text())
        manifest_windows = meta.get("windows", [30])
    if len(manifest_windows) == 1:
        window = manifest_windows[0]
        st.info(f"Lookback Window: **{window} trading days** (fixed for this model)")
    else:
        window = st.select_slider(
            "Lookback Window (trading days)", options=manifest_windows,
            value=manifest_windows[0],
        )

    st.markdown("### 📊 Chart Options")
    show_ma  = st.checkbox("Moving Averages (20d, 50d)", value=True)
    show_vol = st.checkbox("Volume Bars", value=False)

    st.markdown("### 🤖 Model Options")
    use_yolo = st.checkbox("YOLOv8 Bounding Boxes", value=True,
                            help="Draws boxes around detected pattern regions")
    yolo_conf = st.slider("YOLOv8 confidence threshold", 0.20, 0.90, 0.35, 0.05)

    st.divider()
    st.caption(f"Spring 2026 | AI & Deep Learning Course · Model: `{MODEL_NAME}` · Classes: {len(CLASSES)}")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for **educational purposes only**. "
        "Pattern detections are not financial advice. Past chart patterns do "
        "not guarantee future performance."
    )

# ── Main Area ────────────────────────────────────────────────────────
st.title("📈 AI Stock Chart Pattern Recognition")
st.markdown(
    f"Real-time pattern detection — **{MODEL_NAME}** classifier "
    f"({len(CLASSES)} classes) + **YOLOv8** localization."
)
st.divider()

col_btn, col_status = st.columns([1, 4])
with col_btn:
    run = st.button("🔍 Analyze Chart", type="primary", use_container_width=True)
with col_status:
    if ticker:
        st.info(
            f"Ready to analyze **{ticker}** · "
            f"**{window}-day** window · "
            f"MA overlays: {'ON' if show_ma else 'OFF'}"
        )

# ── Analysis Pipeline ────────────────────────────────────────────────
if run:
    t_total_start = time.time()

    with st.status("Pulling live data from Yahoo Finance...", expanded=False):
        # A generous multiplier ensures enough trading days after tail().
        period_map = {30: "3mo", 60: "6mo", 90: "9mo"}
        period = period_map.get(window, "6mo")
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False,
            )
        except Exception as e:
            st.error(f"Failed to fetch data for **{ticker}**: {e}")
            st.stop()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        ohlcv = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().tail(window)
        st.write(f"Fetched {len(ohlcv)} rows for {ticker}")

    if len(ohlcv) < 15:
        st.error(
            f"Not enough data for **{ticker}** ({len(ohlcv)} rows). "
            "Check the ticker or try a longer window."
        )
        st.stop()

    with st.status("Generating candlestick chart...", expanded=False):
        chart_img       = generate_chart(ohlcv, ticker, show_ma, show_vol)
        model_chart_img = generate_model_chart(ohlcv)
        st.write("Charts generated (display: 8×5 in, model input: 224×224 px)")

    cnn_model, cnn_err = load_cnn()
    pred_idx, probs = None, None
    pred_class, confidence = None, None

    if cnn_model is None:
        st.warning(f"⚠️ CNN model not available: {cnn_err}")
        st.info("Train the model first: `python src/train_cnn.py`")
    else:
        with st.status("Running CNN pattern classification...", expanded=False):
            pred_idx, probs = cnn_inference(cnn_model, model_chart_img)
            pred_class  = CLASSES[pred_idx]
            confidence  = float(probs[pred_idx])
            st.write(f"Prediction: {pred_class} ({confidence:.1%})")

    # YOLOv8 — run ONCE and reuse (previous bug ran inference twice)
    annotated_img = None
    yolo_results  = None
    if use_yolo:
        yolo_model, yolo_err = load_yolo()
        if yolo_model is None:
            st.warning(f"⚠️ YOLOv8 not available: {yolo_err}")
        else:
            with st.status("Running YOLOv8 pattern localization...", expanded=False):
                yolo_results   = yolo_model(chart_img, conf=yolo_conf, verbose=False)
                ann_arr        = yolo_results[0].plot()
                annotated_img  = Image.fromarray(ann_arr)
                n_boxes = len(yolo_results[0].boxes)
                st.write(f"Detected {n_boxes} pattern region(s) at conf >= {yolo_conf}")

    t_total = time.time() - t_total_start

    # ── Results Display ──────────────────────────────────────────────
    st.divider()

    if pred_idx is not None:
        color, _ = meta_for(pred_class)
        st.markdown(f"""
        <div style='background:{color};padding:18px 28px;border-radius:14px;
                    text-align:center;color:white;margin:8px 0 16px 0;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
            <div style='font-size:30px;font-weight:bold;letter-spacing:1px;'>
                {pred_class.replace('_', ' ').title()}
            </div>
            <div style='font-size:16px;opacity:0.9;margin-top:4px;'>
                Confidence: {confidence:.1%} &nbsp;·&nbsp;
                ⏱ {t_total:.2f}s end-to-end
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, desc = meta_for(pred_class)
        st.info(f"📖 {desc}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"Candlestick Chart — {ticker}")
        st.image(chart_img, use_column_width=True)

    with col_r:
        if annotated_img is not None:
            st.subheader("YOLOv8 Pattern Localization")
            st.image(annotated_img, use_column_width=True)
            boxes = yolo_results[0].boxes
            if len(boxes) > 0:
                st.markdown("**Detected by YOLOv8:**")
                seen = set()
                for box in boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = yolo_results[0].names.get(cls_id, f"Class {cls_id}")
                    c        = float(box.conf[0])
                    if cls_name not in seen:
                        _, desc = meta_for(cls_name)
                        st.markdown(f"- **{cls_name}** ({c:.0%}) — {desc}")
                        seen.add(cls_name)
            else:
                st.caption("No patterns localised by YOLOv8 in this window.")
        elif pred_idx is not None:
            st.subheader("Confidence Across All Classes")
            prob_data = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            st.bar_chart(prob_data, color="#3498db")

    if probs is not None:
        st.subheader("Confidence Breakdown")
        # Show up to 4 per row for N-class UIs.
        per_row = min(4, len(CLASSES))
        rows    = (len(CLASSES) + per_row - 1) // per_row
        idx     = 0
        for _ in range(rows):
            cols_row = st.columns(per_row)
            for j in range(per_row):
                if idx >= len(CLASSES):
                    break
                cls = CLASSES[idx]
                col = cols_row[j]
                delta_color = "normal" if idx == pred_idx else "off"
                col.metric(
                    label=cls.replace('_', ' ').title(),
                    value=f"{float(probs[idx]):.1%}",
                    delta="← detected" if idx == pred_idx else None,
                    delta_color=delta_color,
                )
                idx += 1

    st.subheader("Price Summary")
    price_cols = st.columns(4)
    close_vals = ohlcv['Close'].values
    price_cols[0].metric("Current Close", f"${float(close_vals[-1]):.2f}")
    price_cols[1].metric("Window High",   f"${float(ohlcv['High'].max()):.2f}")
    price_cols[2].metric("Window Low",    f"${float(ohlcv['Low'].min()):.2f}")
    pct_chg = (float(close_vals[-1]) - float(close_vals[0])) / float(close_vals[0]) * 100
    price_cols[3].metric("Window Return", f"{pct_chg:+.2f}%",
                           delta_color="normal" if pct_chg >= 0 else "inverse")

    with st.expander("📊 View Raw OHLCV Data"):
        display_df = ohlcv.copy()
        for col_name in ['Open', 'High', 'Low', 'Close']:
            display_df[col_name] = display_df[col_name].map('${:.2f}'.format)
        display_df['Volume'] = ohlcv['Volume'].map('{:,.0f}'.format)
        st.dataframe(display_df.tail(15), use_container_width=True)

    buf = io.BytesIO()
    chart_img.save(buf, format='PNG')
    st.download_button(
        label="⬇️ Download Chart", data=buf.getvalue(),
        file_name=f"{ticker}_{window}d_chart.png", mime="image/png",
    )

    lat_ok = t_total <= 3.0
    st.metric(
        "⏱ End-to-End Latency", f"{t_total:.2f}s",
        delta="✓ Under 3s KPI target" if lat_ok else "⚠ Over 3s KPI target",
        delta_color="normal" if lat_ok else "inverse",
    )

# ── Landing Page ─────────────────────────────────────────────────────
else:
    col_info, col_patterns = st.columns([1, 1])

    with col_info:
        st.markdown("""
        ### How it works
        1. Enter a **stock ticker** in the sidebar
        2. Choose a **lookback window** (if multiple are supported)
        3. Click **Analyze Chart**

        The app pulls live OHLCV data, generates a candlestick chart,
        runs it through a fine-tuned CNN for classification, and
        optionally through **YOLOv8** to draw bounding boxes on the
        detected pattern region.
        """)

        st.markdown("""
        ### Pipeline
        ```
        yfinance → OHLCV data
            ↓
        mplfinance → Candlestick chart (PNG)
            ↓
        CNN classifier → Pattern class + confidence
            ↓
        YOLOv8 → Bounding box on chart region
            ↓
        Streamlit → Live display
        ```
        """)

    with col_patterns:
        st.markdown(f"### Detected Patterns ({len(CLASSES)} classes)")
        for cls in CLASSES:
            color, desc = meta_for(cls)
            st.markdown(f"""
            <div style='border-left: 4px solid {color};
                        padding: 8px 12px; margin: 6px 0;
                        border-radius: 4px; background: #f8f9fa;'>
                <b style='color:{color};'>{cls.replace('_', ' ').title()}</b><br>
                <small>{desc[:160]}...</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='text-align:center; color:#999; font-size:13px;'>
        Built with PyTorch · {MODEL_NAME} · YOLOv8 · mplfinance · Streamlit
        &nbsp;|&nbsp; Group 3 — Images in Finance — Spring 2026
    </div>
    """, unsafe_allow_html=True)
