"""
Group 3 — AI Stock Chart Pattern Recognition
Live Streamlit Dashboard

Features:
  - Real-time OHLCV data via yfinance
  - Candlestick chart generation via mplfinance
  - CNN pattern classification (EfficientNet-B0)
  - YOLOv8 bounding box detection
  - Confidence breakdown across all 6 pattern classes
  - End-to-end latency display

Run locally:  streamlit run app/streamlit_app.py
Deploy:       Streamlit Community Cloud → share.streamlit.io
"""

import io
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

# ── Constants ────────────────────────────────────────────────────────
CLASSES = [
    'Head & Shoulders', 'Double Top', 'Double Bottom',
    'Triangle', 'Wedge', 'No Pattern'
]

PATTERN_COLORS = {
    'Head & Shoulders': '#c0392b',
    'Double Top':       '#e67e22',
    'Double Bottom':    '#27ae60',
    'Triangle':         '#2980b9',
    'Wedge':            '#8e44ad',
    'No Pattern':       '#7f8c8d',
}

PATTERN_DESCRIPTIONS = {
    'Head & Shoulders': (
        '**Bearish reversal signal.** Three peaks where the middle (head) is '
        'the highest and the two outer peaks (shoulders) are at similar levels. '
        'Often signals a transition from an uptrend to a downtrend.'
    ),
    'Double Top': (
        '**Bearish reversal signal.** Two peaks at approximately the same price '
        'level, separated by a moderate trough. Suggests the asset is struggling '
        'to break through resistance and momentum may be weakening.'
    ),
    'Double Bottom': (
        '**Bullish reversal signal.** Two troughs at approximately the same price '
        'level, separated by a moderate peak. Often signals that a downtrend is '
        'ending and a new uptrend may be beginning.'
    ),
    'Triangle': (
        '**Continuation or breakout pattern.** Converging trendlines create a '
        'triangle shape. Ascending triangles are generally bullish, descending '
        'triangles bearish, and symmetrical triangles neutral until a breakout.'
    ),
    'Wedge': (
        '**Reversal pattern.** Both support and resistance lines slope in the '
        'same direction. A rising wedge is typically bearish; a falling wedge '
        'is typically bullish.'
    ),
    'No Pattern': (
        '**No classical pattern detected** in the selected window. The chart '
        'does not match any of the six trained pattern classes with sufficient '
        'confidence. This is a valid and common result.'
    ),
}

POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL',
    'META', 'AMZN', 'JPM', 'V', 'XOM'
]

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Model Loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model...")
def load_cnn():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 6)
        )
        weights_path = MODELS_DIR / "best_cnn.pth"
        model.load_state_dict(
            torch.load(str(weights_path), map_location='cpu'))
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
        weights_path = MODELS_DIR / "best.pt"
        return YOLO(str(weights_path)), None
    except FileNotFoundError:
        return None, "models/best.pt not found. Run src/download_datasets.py first."
    except Exception as e:
        return None, str(e)


def cnn_inference(model, img_pil):
    """Run CNN inference. Returns (predicted_class_idx, probabilities_tensor)."""
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = tf(img_pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return pred_idx, probs


# ── Chart Generation ─────────────────────────────────────────────────
def generate_chart(ohlcv: pd.DataFrame, ticker: str,
                   show_ma: bool, show_volume: bool) -> Image.Image:
    """Generate a candlestick chart and return as PIL Image."""
    add_plots = []
    if show_ma:
        if len(ohlcv) >= 20:
            ma20 = ohlcv['Close'].rolling(20).mean()
            add_plots.append(mpf.make_addplot(
                ma20, color='#3498db', width=1.2, label='MA20'))
        if len(ohlcv) >= 50:
            ma50 = ohlcv['Close'].rolling(50).mean()
            add_plots.append(mpf.make_addplot(
                ma50, color='#e67e22', width=1.2, label='MA50'))

    style = mpf.make_mpf_style(
        base_mpf_style='charles',
        gridstyle='--',
        gridcolor='#ecf0f1',
        rc={'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7}
    )

    fig, _ = mpf.plot(
        ohlcv, type='candle', style=style,
        addplot=add_plots if add_plots else [],
        volume=show_volume, returnfig=True,
        figsize=(8, 5), tight_layout=True,
        title=f'\n{ticker} — {len(ohlcv)}-Day Candlestick'
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).convert('RGB')


# ── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Pattern Detector | Group 3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 13px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
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
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter any valid ticker (e.g. AAPL, TSLA, NVDA)"
    ).upper().strip()

    # Quick-pick buttons
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
    window = st.select_slider(
        "Lookback Window (trading days)",
        options=[30, 60, 90],
        value=30
    )

    st.markdown("### 📊 Chart Options")
    show_ma  = st.checkbox("Moving Averages (20d, 50d)", value=True)
    show_vol = st.checkbox("Volume Bars", value=False)

    st.markdown("### 🤖 Model Options")
    use_yolo = st.checkbox("YOLOv8 Bounding Boxes", value=True,
                            help="Draws boxes around detected pattern regions")

    st.divider()
    st.caption("Spring 2026 | AI & Deep Learning Course")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for **educational purposes only**. "
        "Pattern detections are not financial advice. Past chart patterns "
        "do not guarantee future performance."
    )

# ── Main Area ────────────────────────────────────────────────────────
st.title("📈 AI Stock Chart Pattern Recognition")
st.markdown(
    "Real-time pattern detection using **EfficientNet-B0** (classification) "
    "and **YOLOv8** (localization) trained on 9,000+ annotated chart images."
)
st.divider()

# Analyze button
col_btn, col_status = st.columns([1, 4])
with col_btn:
    run = st.button("🔍 Analyze Chart", type="primary",
                     use_container_width=True)
with col_status:
    if ticker:
        st.info(f"Ready to analyze **{ticker}** · "
                f"**{window}-day** window · "
                f"MA overlays: {'ON' if show_ma else 'OFF'}")

# ── Analysis Pipeline ────────────────────────────────────────────────
if run:
    t_total_start = time.time()

    # 1. Fetch data
    with st.status("Pulling live data from Yahoo Finance...", expanded=False):
        period_map = {30: "3mo", 60: "5mo", 90: "7mo"}
        try:
            raw = yf.download(
                ticker, period=period_map[window],
                interval="1d", auto_adjust=True, progress=False
            )
        except Exception as e:
            st.error(f"Failed to fetch data for **{ticker}**: {e}")
            st.stop()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        ohlcv = raw[['Open','High','Low','Close','Volume']].dropna().tail(window)
        st.write(f"Fetched {len(ohlcv)} rows for {ticker}")

    if len(ohlcv) < 15:
        st.error(
            f"Not enough data for **{ticker}** ({len(ohlcv)} rows). "
            "Check the ticker symbol or try a longer window."
        )
        st.stop()

    # 2. Generate chart
    with st.status("Generating candlestick chart...", expanded=False):
        chart_img = generate_chart(ohlcv, ticker, show_ma, show_vol)
        st.write("Chart generated (224×224 px)")

    # 3. CNN inference
    cnn_model, cnn_err = load_cnn()
    pred_idx, probs    = None, None

    if cnn_model is None:
        st.warning(f"⚠️ CNN model not available: {cnn_err}")
        st.info("Showing chart only. Train the model first: `python src/train_cnn.py`")
    else:
        with st.status("Running CNN pattern classification...", expanded=False):
            pred_idx, probs = cnn_inference(cnn_model, chart_img)
            pred_class  = CLASSES[pred_idx]
            confidence  = float(probs[pred_idx])
            st.write(f"Prediction: {pred_class} ({confidence:.1%})")

    # 4. YOLOv8
    annotated_img = None
    if use_yolo:
        yolo_model, yolo_err = load_yolo()
        if yolo_model is None:
            st.warning(f"⚠️ YOLOv8 not available: {yolo_err}")
        else:
            with st.status("Running YOLOv8 pattern localization...",
                            expanded=False):
                results       = yolo_model(chart_img, conf=0.20, verbose=False)
                ann_arr       = results[0].plot()
                annotated_img = Image.fromarray(ann_arr)
                n_boxes = len(results[0].boxes)
                st.write(f"Detected {n_boxes} pattern region(s)")

    t_total = time.time() - t_total_start

    # ── Results Display ──────────────────────────────────────────────
    st.divider()

    # Prediction banner
    if pred_idx is not None:
        color = PATTERN_COLORS.get(pred_class, '#7f8c8d')
        st.markdown(f"""
        <div style='background:{color};padding:18px 28px;border-radius:14px;
                    text-align:center;color:white;margin:8px 0 16px 0;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
            <div style='font-size:30px;font-weight:bold;letter-spacing:1px;'>
                {pred_class}
            </div>
            <div style='font-size:16px;opacity:0.9;margin-top:4px;'>
                Confidence: {confidence:.1%} &nbsp;·&nbsp;
                ⏱ {t_total:.2f}s end-to-end
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"📖 {PATTERN_DESCRIPTIONS[pred_class]}")

    # Charts side by side
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"Candlestick Chart — {ticker}")
        st.image(chart_img, use_container_width=True)

    with col_r:
        if annotated_img:
            st.subheader("YOLOv8 Pattern Localization")
            st.image(annotated_img, use_container_width=True)
        elif pred_idx is not None:
            st.subheader("Confidence Across All Classes")
            prob_data = {CLASSES[i]: float(probs[i]) for i in range(6)}
            st.bar_chart(prob_data, color="#3498db")

    # Confidence metrics row
    if probs is not None:
        st.subheader("Confidence Breakdown")
        metric_cols = st.columns(6)
        for i, (cls, col) in enumerate(zip(CLASSES, metric_cols)):
            delta_color = "normal" if i == pred_idx else "off"
            col.metric(
                label=cls,
                value=f"{float(probs[i]):.1%}",
                delta="← detected" if i == pred_idx else None,
                delta_color=delta_color
            )

    # Price summary
    st.subheader("Price Summary")
    price_cols = st.columns(4)
    close_vals = ohlcv['Close'].values
    price_cols[0].metric("Current Close",
                           f"${float(close_vals[-1]):.2f}")
    price_cols[1].metric("Window High",
                           f"${float(ohlcv['High'].max()):.2f}")
    price_cols[2].metric("Window Low",
                           f"${float(ohlcv['Low'].min()):.2f}")
    pct_chg = (float(close_vals[-1]) - float(close_vals[0])) / float(close_vals[0]) * 100
    price_cols[3].metric("Window Return",
                           f"{pct_chg:+.2f}%",
                           delta_color="normal" if pct_chg >= 0 else "inverse")

    # Raw data
    with st.expander("📊 View Raw OHLCV Data"):
        display_df = ohlcv.copy()
        for col_name in ['Open','High','Low','Close']:
            display_df[col_name] = display_df[col_name].map('${:.2f}'.format)
        display_df['Volume'] = ohlcv['Volume'].map('{:,.0f}'.format)
        st.dataframe(display_df.tail(15), use_container_width=True)

    # Download chart
    buf = io.BytesIO()
    chart_img.save(buf, format='PNG')
    st.download_button(
        label="⬇️ Download Chart",
        data=buf.getvalue(),
        file_name=f"{ticker}_{window}d_chart.png",
        mime="image/png"
    )

    # Latency KPI
    lat_ok = t_total <= 3.0
    st.metric(
        "⏱ End-to-End Latency",
        f"{t_total:.2f}s",
        delta="✓ Under 3s KPI target" if lat_ok else "⚠ Over 3s KPI target",
        delta_color="normal" if lat_ok else "inverse"
    )

# ── Landing Page (no run yet) ─────────────────────────────────────────
else:
    col_info, col_patterns = st.columns([1, 1])

    with col_info:
        st.markdown("""
        ### How it works
        1. Enter a **stock ticker** in the sidebar
        2. Choose a **lookback window** (30 / 60 / 90 days)
        3. Click **Analyze Chart**

        The app pulls live OHLCV data, generates a candlestick chart,
        runs it through a fine-tuned **EfficientNet-B0** CNN for
        classification, and optionally through **YOLOv8** to draw
        bounding boxes on the detected pattern region.
        """)

        st.markdown("""
        ### Pipeline
        ```
        yfinance → OHLCV data
            ↓
        mplfinance → Candlestick chart (PNG)
            ↓
        EfficientNet-B0 → Pattern class + confidence
            ↓
        YOLOv8 → Bounding box on chart region
            ↓
        Streamlit → Live display
        ```
        """)

    with col_patterns:
        st.markdown("### Detected Patterns")
        for cls, desc in PATTERN_DESCRIPTIONS.items():
            color = PATTERN_COLORS[cls]
            st.markdown(f"""
            <div style='border-left: 4px solid {color};
                        padding: 8px 12px; margin: 6px 0;
                        border-radius: 4px; background: #f8f9fa;'>
                <b style='color:{color};'>{cls}</b><br>
                <small>{desc.replace("**","")[:120]}...</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#999; font-size:13px;'>
        Built with PyTorch · EfficientNet-B0 · YOLOv8 · mplfinance · Streamlit
        &nbsp;|&nbsp; Group 3 — Images in Finance — Spring 2026
    </div>
    """, unsafe_allow_html=True)
