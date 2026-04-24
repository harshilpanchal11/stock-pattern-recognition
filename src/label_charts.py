"""
Algorithmic Chart Pattern Labeler
Detects Double Top and Double Bottom patterns in our mplfinance-generated
charts using peak/trough detection on the underlying OHLCV price data.

This creates a fully consistent dataset (all images from same mplfinance
pipeline) which eliminates the domain mismatch from mixing Kaggle images
with our generated charts.

Labels saved to: data/cnn_ready_v2/
Run: python src/label_charts.py
"""

import shutil
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CHART_DIR = Path(__file__).parent.parent / "data" / "charts"
OUT_DIR   = Path(__file__).parent.parent / "data" / "cnn_ready_v2"

WINDOW    = 30      # only use 30-day windows for consistency
STEP      = 10

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END   = pd.Timestamp("2024-12-31")

# ── Pattern-detection thresholds ─────────────────────────────────────
# Tightened from the previous values; the old thresholds produced noisy
# labels that capped accuracy at ~0.60 F1.  The new values require a
# clearer, more geometrically-convincing pattern before a window is
# accepted as positive.  Everything that does not qualify is now treated
# as *ambiguous* and is excluded from the training set entirely (NOT
# silently dumped into no_pattern, which was the main source of bias).
PEAK_SIMILARITY    = 0.025  # peaks must be within 2.5% of each other
TROUGH_DEPTH       = 0.05   # trough must be >= 5% below peak average
MIN_PEAK_SEP       = 7      # peaks must be >= 7 bars apart
MIN_PATTERN_SPAN   = 0.40   # pattern must span >= 40% of the window
MAX_PATTERN_SPAN   = 0.95   # and <= 95% of it (reject end-to-end spans)
NO_PATTERN_MAX_PEAK_SIM = 0.10   # to be safe-no-pattern, the two top
                                  # peaks must differ by >=10% (so we
                                  # don't mis-label near-doubles as
                                  # no_pattern)


def find_peaks(series: np.ndarray, min_sep: int = 3) -> np.ndarray:
    """Find local maxima with minimum separation."""
    peaks = []
    for i in range(1, len(series) - 1):
        if series[i] >= series[i-1] and series[i] >= series[i+1]:
            if not peaks or i - peaks[-1] >= min_sep:
                peaks.append(i)
            elif series[i] > series[peaks[-1]]:
                peaks[-1] = i
    return np.array(peaks)


def find_troughs(series: np.ndarray, min_sep: int = 3) -> np.ndarray:
    """Find local minima with minimum separation."""
    return find_peaks(-series, min_sep)


def is_double_top(close: np.ndarray) -> bool:
    """
    Double Top: two peaks at similar height with a trough between them.
    - Two highest peaks within PEAK_SIMILARITY of each other
    - Trough between peaks at least TROUGH_DEPTH below peak average
    - Peaks separated by at least MIN_PEAK_SEP bars
    - Pattern span in [MIN_PATTERN_SPAN, MAX_PATTERN_SPAN] of the window
    """
    peaks = find_peaks(close, min_sep=MIN_PEAK_SEP)
    if len(peaks) < 2:
        return False

    peak_vals = close[peaks]
    top2_idx  = np.argsort(peak_vals)[-2:]
    p1, p2    = sorted(peaks[top2_idx])
    v1, v2    = close[p1], close[p2]

    avg_peak = (v1 + v2) / 2
    if abs(v1 - v2) / avg_peak > PEAK_SIMILARITY:
        return False
    if p2 - p1 < MIN_PEAK_SEP:
        return False

    between    = close[p1:p2+1]
    min_trough = between.min()
    if (avg_peak - min_trough) / avg_peak < TROUGH_DEPTH:
        return False

    span = (p2 - p1) / len(close)
    if span < MIN_PATTERN_SPAN or span > MAX_PATTERN_SPAN:
        return False

    return True


def is_double_bottom(close: np.ndarray) -> bool:
    """
    Double Bottom: mirror of is_double_top on -close.
    """
    troughs = find_troughs(close, min_sep=MIN_PEAK_SEP)
    if len(troughs) < 2:
        return False

    trough_vals = close[troughs]
    bot2_idx    = np.argsort(trough_vals)[:2]
    t1, t2      = sorted(troughs[bot2_idx])
    v1, v2      = close[t1], close[t2]

    avg_trough = (v1 + v2) / 2
    if abs(v1 - v2) / avg_trough > PEAK_SIMILARITY:
        return False
    if t2 - t1 < MIN_PEAK_SEP:
        return False

    between  = close[t1:t2+1]
    max_peak = between.max()
    if (max_peak - avg_trough) / avg_trough < TROUGH_DEPTH:
        return False

    span = (t2 - t1) / len(close)
    if span < MIN_PATTERN_SPAN or span > MAX_PATTERN_SPAN:
        return False

    return True


def is_safe_no_pattern(close: np.ndarray) -> bool:
    """
    Stricter 'no-pattern' definition — a window is only labelled
    no_pattern when we are reasonably confident it is NOT a near-miss
    Double-Top or Double-Bottom.  This cuts label noise.
    """
    peaks = find_peaks(close, min_sep=MIN_PEAK_SEP)
    troughs = find_troughs(close, min_sep=MIN_PEAK_SEP)

    # Reject as "safe no_pattern" if the top-2 peaks are too similar.
    if len(peaks) >= 2:
        top2 = np.sort(close[peaks])[-2:]
        if abs(top2[0] - top2[1]) / ((top2[0] + top2[1]) / 2) \
                < NO_PATTERN_MAX_PEAK_SIM:
            return False
    # Same for bottom-2 troughs.
    if len(troughs) >= 2:
        bot2 = np.sort(close[troughs])[:2]
        if abs(bot2[0] - bot2[1]) / ((bot2[0] + bot2[1]) / 2) \
                < NO_PATTERN_MAX_PEAK_SIM:
            return False
    return True


def get_split(date: pd.Timestamp) -> str:
    if date <= TRAIN_END: return "train"
    if date <= VAL_END:   return "val"
    return "test"


def label_all_charts():
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        print("No Parquet files found. Run src/ingest.py first.")
        return

    # Collect all labeled paths before copying so we can balance
    labeled_pool    = {"double_top": {}, "double_bottom": {}}  # split → list of paths
    no_pattern_pool = []

    for split in ["train", "val", "test"]:
        labeled_pool["double_top"][split]    = []
        labeled_pool["double_bottom"][split] = []

    print(f"Labeling charts from {len(parquet_files)} tickers...")

    for pf in parquet_files:
        ticker = pf.stem
        df     = pd.read_parquet(pf)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna()

        for i in range(0, len(df) - WINDOW, STEP):
            chunk    = df.iloc[i : i + WINDOW]
            end_date = chunk.index[-1]
            split    = get_split(end_date)
            close    = chunk['Close'].values

            fname    = f"{ticker}_{WINDOW}d_{end_date.strftime('%Y%m%d')}.png"
            src_path = CHART_DIR / split / "unlabeled" / fname
            if not src_path.exists():
                continue

            dt = is_double_top(close)
            db = is_double_bottom(close)

            if dt and db:
                # Ambiguous — discard (was: silently labelled double_top)
                continue
            if dt:
                labeled_pool["double_top"][split].append(src_path)
            elif db:
                labeled_pool["double_bottom"][split].append(src_path)
            elif is_safe_no_pattern(close):
                no_pattern_pool.append((src_path, split))
            # else: near-miss — drop from dataset rather than mis-label

    # ── Balance: cap double_top at double_bottom count per split ──────
    counts = Counter()
    for split in ["train", "val", "test"]:
        dt_paths = labeled_pool["double_top"][split]
        db_paths = labeled_pool["double_bottom"][split]

        # Shuffle then cap the majority class at minority class size
        random.shuffle(dt_paths)
        random.shuffle(db_paths)
        cap = min(len(dt_paths), len(db_paths))

        for label, paths in [("double_top", dt_paths[:cap]),
                              ("double_bottom", db_paths[:cap])]:
            dest = OUT_DIR / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for src_path in paths:
                shutil.copy2(src_path, dest / src_path.name)
            counts[f"{split}/{label}"] = cap

    # Balance no_pattern: sample ~2x the pattern count per split
    for split in ["train", "val", "test"]:
        pattern_count = sum(
            v for k, v in counts.items()
            if k.startswith(split) and "no_pattern" not in k
        )
        target = max(pattern_count, 50)   # 1× pattern count (not 2×)
        pool   = [(p, s) for p, s in no_pattern_pool if s == split]
        random.shuffle(pool)
        sampled = pool[:target]

        dest = OUT_DIR / split / "no_pattern"
        dest.mkdir(parents=True, exist_ok=True)
        for src_path, _ in sampled:
            shutil.copy2(src_path, dest / src_path.name)
        counts[f"{split}/no_pattern"] = len(sampled)

    # Print summary
    print(f"\n── Labeling Summary ────────────────────────────")
    total = 0
    for split in ["train", "val", "test"]:
        for label in ["double_top", "double_bottom", "no_pattern"]:
            key = f"{split}/{label}"
            n   = counts.get(key, 0)
            total += n
            print(f"  {split:<6} / {label:<20} {n} images")
    print(f"  {'TOTAL':<28} {total} images")
    print(f"\n  Output: {OUT_DIR}")
    print("  Next: python src/train_cnn.py --data_dir data/cnn_ready_v2")


if __name__ == "__main__":
    label_all_charts()
