"""
Stage 2 — Chart Generation
Converts OHLCV Parquet files into 224x224 candlestick chart PNGs
using sliding windows of 30, 60, and 90 trading days.

Run: python src/generate_charts.py
     python src/generate_charts.py --window 30 --step 5 --demo   (fast demo mode)
"""

import argparse
import mplfinance as mpf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CHART_DIR = Path(__file__).parent.parent / "data" / "charts"

WINDOWS   = [30, 60, 90]   # trading day window sizes
STEP      = 10             # default slide step (use 5 for more images)

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END   = pd.Timestamp("2024-12-31")
# test = anything after VAL_END

MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style='charles',
    gridstyle='--',
    gridcolor='#ecf0f1',
    rc={
        'figure.figsize': (2.24, 2.24),
        'figure.dpi': 100,
        'axes.labelsize': 0,
        'xtick.labelsize': 0,
        'ytick.labelsize': 0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)

# ── Helpers ──────────────────────────────────────────────────────────
def get_split(date: pd.Timestamp) -> str:
    if date <= TRAIN_END: return "train"
    if date <= VAL_END:   return "val"
    return "test"


def make_add_plots(chunk: pd.DataFrame) -> list:
    add_plots = []
    if len(chunk) >= 20:
        ma20 = chunk['Close'].rolling(20).mean()
        add_plots.append(mpf.make_addplot(ma20, color='#3498db', width=0.8))
    if len(chunk) >= 50:
        ma50 = chunk['Close'].rolling(50).mean()
        add_plots.append(mpf.make_addplot(ma50, color='#e67e22', width=0.8))
    return add_plots


def save_chart(chunk: pd.DataFrame, out_path: Path, with_indicators: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    add_plots = make_add_plots(chunk) if with_indicators else []

    try:
        fig, _ = mpf.plot(
            chunk, type='candle', style=MPF_STYLE,
            addplot=add_plots if add_plots else [],
            volume=False, returnfig=True, tight_layout=True,
        )
        fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except Exception as e:
        plt.close('all')
        return False


# ── Main ─────────────────────────────────────────────────────────────
def generate(windows: list, step: int, demo_mode: bool = False):
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        print("No Parquet files found in data/raw/. Run src/ingest.py first.")
        return

    print(f"Found {len(parquet_files)} tickers | "
          f"Windows: {windows} | Step: {step} | Demo: {demo_mode}")

    total_saved = 0
    total_skipped = 0

    for pf in tqdm(parquet_files, desc="Tickers"):
        ticker = pf.stem
        df = pd.read_parquet(pf)

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].dropna()

        for window in windows:
            if len(df) < window:
                continue

            ticker_count = 0
            for i in range(0, len(df) - window, step):
                chunk    = df.iloc[i : i + window].copy()
                end_date = chunk.index[-1]
                split    = get_split(end_date)

                # In demo mode only generate test split (live data)
                if demo_mode and split != "test":
                    continue

                # Images land in split/unlabeled/ until CNN labels them
                folder   = CHART_DIR / split / "unlabeled"
                fname    = f"{ticker}_{window}d_{end_date.strftime('%Y%m%d')}.png"
                out_path = folder / fname

                if out_path.exists():
                    total_skipped += 1
                    continue

                ok = save_chart(chunk, out_path)
                if ok:
                    total_saved += 1
                    ticker_count += 1

                # In demo mode cap at 5 images per ticker/window
                if demo_mode and ticker_count >= 5:
                    break

    print(f"\n── Chart generation complete ───────────────────")
    print(f"  New images saved: {total_saved}")
    print(f"  Already existed:  {total_skipped}")
    print(f"  Output directory: {CHART_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, nargs="+", default=WINDOWS,
                        help="Window sizes in trading days")
    parser.add_argument("--step",   type=int, default=STEP,
                        help="Slide step in days")
    parser.add_argument("--demo",   action="store_true",
                        help="Demo mode: only generate a small subset quickly")
    args = parser.parse_args()
    generate(args.window, args.step, args.demo)
