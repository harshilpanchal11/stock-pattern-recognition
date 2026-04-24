"""
Stage 3b (NEW) — Synthetic multi-class pattern generator.

Motivation
----------
The two HuggingFace multi-class datasets we tried to pull (foduucom and
rishi-dua) both 404'd.  Rather than wait for a third-party dataset, we
synthesise OHLCV series that *exactly* match each classical pattern and
render them via the same mplfinance style as our real-data charts.

Benefits vs. external datasets
  - Zero external dependencies / no credentials required.
  - Perfect labels (no weak-label noise).
  - No domain shift (same renderer as live Streamlit inference).
  - Unlimited samples, balanced classes.

Classes (7 total)
-----------------
  1. double_top
  2. double_bottom
  3. head_and_shoulders_top
  4. head_and_shoulders_bottom
  5. ascending_triangle
  6. descending_triangle
  7. no_pattern

Each class is generated from a parametric template plus Gaussian noise,
sampled with a fixed SEED for reproducibility.  The OHLCV bars are then
rendered to 224x224 PNGs using mplfinance (style 'charles', MA20/MA50)
and routed into train/val/test via a deterministic hash split (70/15/15).

Output
------
  data/cnn_clean/{train,val,test}/<class>/<class>_<id>.png
  data/cnn_clean/manifest.json
  models/classes.json   (so Streamlit + train_cnn stay in sync)

Run
---
    # Default: 800 samples per class (~30-45 min on Apple Silicon)
    python src/generate_synthetic_patterns.py

    # Fast smoke test
    python src/generate_synthetic_patterns.py --n-per-class 40

    # Headless / CI
    python src/generate_synthetic_patterns.py --n-per-class 200 --workers 4
"""

import argparse
import hashlib
import json
import os
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# mplfinance must be imported *after* matplotlib backend is set.
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "cnn_clean"
MODELS_DIR   = PROJECT_ROOT / "models"

CLASSES = [
    "double_top",
    "double_bottom",
    "head_and_shoulders_top",
    "head_and_shoulders_bottom",
    "ascending_triangle",
    "descending_triangle",
    "no_pattern",
]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test takes the remainder (0.15)

# Rendering — MUST match generate_charts.py AND streamlit_app.py's
# generate_model_chart() exactly.  Any drift here re-introduces the
# domain-shift bug (the reason we pivoted to hybrid training).
CANONICAL_STYLE = mpf.make_mpf_style(
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
    },
)

# Bar-count window used by the rest of the pipeline.  generate_charts.py
# renders 30/60/90-day windows — 60 is the canonical middle.  Keep in
# sync with classes.json so Streamlit defaults to the right window.
WINDOW = 60


# ── Deterministic split ──────────────────────────────────────────────
def split_from_hash(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) / (1 << 128)
    if h < TRAIN_FRAC:               return "train"
    if h < TRAIN_FRAC + VAL_FRAC:    return "val"
    return "test"


# ── OHLCV helpers ────────────────────────────────────────────────────
def _close_to_ohlcv(close: np.ndarray,
                    start_price: float = 100.0,
                    base_vol: float = 1e6,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Convert a close-price trajectory to a plausible OHLCV DataFrame with
    small intrabar noise and realistic volume proportional to |Δclose|.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(close)
    noise = rng.normal(0, 0.004, size=(n, 2))  # open/close wiggle
    opens = close * (1 + noise[:, 0] * 0.3)
    # high = max(open,close) * (1 + small positive), low similarly.
    wick_up   = np.abs(rng.normal(0, 0.006, size=n))
    wick_dn   = np.abs(rng.normal(0, 0.006, size=n))
    highs = np.maximum(opens, close) * (1 + wick_up)
    lows  = np.minimum(opens, close) * (1 - wick_dn)

    # Volume: baseline + spike proportional to abs daily return
    rets = np.diff(close, prepend=close[0]) / close
    vol  = base_vol * (1 + 4 * np.abs(rets)) * rng.uniform(0.6, 1.4, size=n)

    end   = datetime(2025, 1, 1)
    idx   = pd.bdate_range(end=end, periods=n)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": close, "Volume": vol.astype(int)},
        index=idx,
    )
    df.index.name = "Date"
    return df


def _smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


def _baseline_drift(n: int, rng: np.random.Generator, amp: float = 0.04) -> np.ndarray:
    """Mild directional drift so bars aren't perfectly flat between events."""
    slope = rng.uniform(-amp, amp)
    line  = np.linspace(0, slope, n)
    return line


# ── Pattern templates ────────────────────────────────────────────────
# Each generator returns a *close* price series of length WINDOW.

def gen_double_top(rng: np.random.Generator) -> np.ndarray:
    n = WINDOW
    t = np.linspace(0, 1, n)
    peak_h = rng.uniform(0.10, 0.18)          # 10-18 % peak heights
    valley = rng.uniform(0.035, 0.070)        # dip between the two peaks
    asymm  = rng.uniform(-0.012, 0.012)       # small peak height asymmetry
    p1 = rng.uniform(0.18, 0.30)
    p2 = rng.uniform(0.65, 0.80)
    vp = rng.uniform(p1 + 0.15, p2 - 0.15)

    base = _baseline_drift(n, rng, amp=0.02) + 1.0
    bump = lambda c, w: np.exp(-((t - c) / w) ** 2)
    curve = (
        base
        + peak_h * bump(p1, 0.08)
        + (peak_h + asymm) * bump(p2, 0.08)
        - valley * bump(vp, 0.05)
    )
    # Breakdown after the 2nd peak
    curve[int(p2 * n):] -= np.linspace(0, 0.04, n - int(p2 * n))
    curve += rng.normal(0, 0.004, n)
    return _smooth(curve, 2) * 100.0


def gen_double_bottom(rng: np.random.Generator) -> np.ndarray:
    # Mirror of double_top
    s = gen_double_top(rng)
    return (2 * s[0] - s)


def gen_head_and_shoulders_top(rng: np.random.Generator) -> np.ndarray:
    n = WINDOW
    t = np.linspace(0, 1, n)
    shoulder_h = rng.uniform(0.07, 0.11)
    head_h     = shoulder_h + rng.uniform(0.04, 0.07)
    dip        = rng.uniform(0.02, 0.04)

    p_ls = rng.uniform(0.15, 0.25)
    p_h  = rng.uniform(0.42, 0.55)
    p_rs = rng.uniform(0.70, 0.82)
    d1   = (p_ls + p_h) / 2
    d2   = (p_h + p_rs) / 2

    base = _baseline_drift(n, rng, amp=0.02) + 1.0
    bump = lambda c, w: np.exp(-((t - c) / w) ** 2)
    curve = (
        base
        + shoulder_h * bump(p_ls, 0.065)
        + head_h    * bump(p_h,  0.070)
        + shoulder_h * bump(p_rs, 0.065)
        - dip * bump(d1, 0.045)
        - dip * bump(d2, 0.045)
    )
    # Neckline break / selloff
    curve[int(p_rs * n):] -= np.linspace(0, 0.05, n - int(p_rs * n))
    curve += rng.normal(0, 0.004, n)
    return _smooth(curve, 2) * 100.0


def gen_head_and_shoulders_bottom(rng: np.random.Generator) -> np.ndarray:
    s = gen_head_and_shoulders_top(rng)
    return (2 * s[0] - s)


def gen_ascending_triangle(rng: np.random.Generator) -> np.ndarray:
    """
    Flat resistance line on top, rising support line from below.
    Close oscillates between them with a shrinking amplitude.
    """
    n = WINDOW
    t = np.linspace(0, 1, n)
    resistance = 1.10 + rng.uniform(-0.01, 0.01)
    support    = np.linspace(0.95, resistance - 0.01, n) + rng.uniform(-0.005, 0.005)
    # Decaying sinusoid between the two bands
    freq = rng.uniform(3.0, 5.0)
    amp  = (resistance - support) * 0.50
    mid  = (resistance + support) / 2
    curve = mid + amp * np.cos(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    curve = np.clip(curve, support, resistance)
    # Optional breakout in the last 10 %
    if rng.random() < 0.6:
        brk = int(0.9 * n)
        curve[brk:] = np.linspace(curve[brk - 1], resistance + 0.04, n - brk)
    curve += rng.normal(0, 0.003, n)
    return _smooth(curve, 2) * 100.0


def gen_descending_triangle(rng: np.random.Generator) -> np.ndarray:
    """
    Flat support line at the bottom, falling resistance line from above.
    """
    n = WINDOW
    t = np.linspace(0, 1, n)
    support    = 0.95 + rng.uniform(-0.01, 0.01)
    resistance = np.linspace(1.10, support + 0.01, n) + rng.uniform(-0.005, 0.005)
    freq = rng.uniform(3.0, 5.0)
    amp  = (resistance - support) * 0.50
    mid  = (resistance + support) / 2
    curve = mid + amp * np.cos(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    curve = np.clip(curve, support, resistance)
    if rng.random() < 0.6:
        brk = int(0.9 * n)
        curve[brk:] = np.linspace(curve[brk - 1], support - 0.04, n - brk)
    curve += rng.normal(0, 0.003, n)
    return _smooth(curve, 2) * 100.0


def gen_no_pattern(rng: np.random.Generator) -> np.ndarray:
    """
    Random walk / mild trend / sideways with noise — nothing classifiable.
    """
    n = WINDOW
    mode = rng.choice(["walk", "trend", "sideways"])
    if mode == "walk":
        steps = rng.normal(0, 0.012, n).cumsum()
        curve = 1.0 + steps
    elif mode == "trend":
        slope = rng.uniform(-0.15, 0.15)
        curve = 1.0 + np.linspace(0, slope, n) + rng.normal(0, 0.008, n)
    else:  # sideways
        curve = 1.0 + rng.normal(0, 0.01, n)
    return _smooth(curve, 2) * 100.0


GENERATORS = {
    "double_top":                 gen_double_top,
    "double_bottom":               gen_double_bottom,
    "head_and_shoulders_top":      gen_head_and_shoulders_top,
    "head_and_shoulders_bottom":   gen_head_and_shoulders_bottom,
    "ascending_triangle":          gen_ascending_triangle,
    "descending_triangle":         gen_descending_triangle,
    "no_pattern":                  gen_no_pattern,
}


# ── Rendering ────────────────────────────────────────────────────────
def render_chart(df: pd.DataFrame, out_path: Path):
    """
    Render a candlestick chart that is PIXEL-COMPATIBLE with:
      - data/cnn_ready_v2/ (produced by generate_charts.py)
      - inference charts produced by streamlit_app.py::generate_model_chart

    This is the domain-shift fix: the previous synthetic renderer used
    `axisoff=True` (no gridlines) and default mav colors, which was
    visually distinct from both training and inference real charts.
    """
    add_plots = []
    if len(df) >= 20:
        ma20 = df['Close'].rolling(20).mean()
        add_plots.append(mpf.make_addplot(ma20, color='#3498db', width=0.8))
    if len(df) >= 50:
        ma50 = df['Close'].rolling(50).mean()
        add_plots.append(mpf.make_addplot(ma50, color='#e67e22', width=0.8))

    try:
        fig, _ = mpf.plot(
            df, type='candle', style=CANONICAL_STYLE,
            addplot=add_plots if add_plots else [],
            volume=False, returnfig=True, tight_layout=True,
        )
        fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close("all")


# ── Main ─────────────────────────────────────────────────────────────
def build_one(cls: str, idx: int, rng: np.random.Generator) -> Path | None:
    close = GENERATORS[cls](rng)
    df    = _close_to_ohlcv(close, rng=rng)
    stem  = f"{cls}_{idx:05d}"
    split = split_from_hash(stem)
    out   = OUT_DIR / split / cls / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_chart(df, out)
        return out
    except Exception as e:
        print(f"    ! render failed for {stem}: {e}")
        return None


def main(n_per_class: int, workers: int):
    # Wipe previous synthetic output (but keep foduucom crops if any).
    # Tolerate files that can't be deleted (locked / permissioned) —
    # they'll be overwritten by the new render anyway.
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = OUT_DIR / split / cls
            if d.exists():
                for f in d.glob(f"{cls}_*.png"):
                    try:
                        f.unlink(missing_ok=True)
                    except PermissionError:
                        pass

    total = len(CLASSES) * n_per_class
    print(f"── Generating {total} synthetic chart images ──")
    print(f"   classes:     {len(CLASSES)}")
    print(f"   per class:   {n_per_class}")
    print(f"   output:      {OUT_DIR}")

    counts = {s: {c: 0 for c in CLASSES} for s in ("train", "val", "test")}

    # Serial path (default).  mplfinance isn't fork-safe; use workers
    # only if the user explicitly requests it and knows the caveats.
    if workers <= 1:
        rng_master = np.random.default_rng(SEED)
        for cls in CLASSES:
            print(f"\n  [{cls}]")
            for i in range(n_per_class):
                # Derive per-sample rng so seeding is stable per-class.
                seed_i = int(rng_master.integers(0, 2**31 - 1))
                rng_i  = np.random.default_rng(seed_i)
                out = build_one(cls, i, rng_i)
                if out is not None:
                    counts[split_from_hash(f"{cls}_{i:05d}")][cls] += 1
                if (i + 1) % max(1, n_per_class // 10) == 0:
                    print(f"     {i + 1}/{n_per_class}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"   workers:     {workers}  (processes)")
        rng_master = np.random.default_rng(SEED)
        jobs = []
        for cls in CLASSES:
            for i in range(n_per_class):
                seed_i = int(rng_master.integers(0, 2**31 - 1))
                jobs.append((cls, i, seed_i))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_worker, cls, i, seed_i)
                for (cls, i, seed_i) in jobs
            ]
            done = 0
            for fut in as_completed(futures):
                cls, i, ok = fut.result()
                if ok:
                    counts[split_from_hash(f"{cls}_{i:05d}")][cls] += 1
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"   {done}/{total}")

    # Manifest for training + streamlit
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    classes_sorted = sorted(CLASSES)
    (MODELS_DIR / "classes.json").write_text(json.dumps(
        {"classes": classes_sorted, "model": "efficientnet_v2_s",
         "source": "synthetic", "seed": SEED,
         "windows": [WINDOW]},   # Streamlit reads this to set the lookback
        indent=2,
    ))
    (OUT_DIR / "manifest.json").write_text(json.dumps(
        {"classes": classes_sorted, "seed": SEED,
         "n_per_class": n_per_class, "source": "synthetic",
         "splits": counts},
        indent=2,
    ))

    # Summary
    print("\n── Summary ──")
    for split in ("train", "val", "test"):
        total_s = sum(counts[split].values())
        print(f"  {split}:  {total_s} images")
        for cls in classes_sorted:
            print(f"     {cls:<34} {counts[split][cls]}")

    print(f"\n  classes.json: {MODELS_DIR / 'classes.json'}")
    print(f"  manifest:     {OUT_DIR / 'manifest.json'}")
    print("\nNext:")
    print("  python tests/test_leakage.py")
    print("  python src/train_cnn.py --data_dir data/cnn_clean "
          "--model efficientnet_v2_s --epochs 30")


def _worker(cls: str, i: int, seed: int):
    rng = np.random.default_rng(seed)
    out = build_one(cls, i, rng)
    return (cls, i, out is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=800,
                        help="Samples per class (default 800).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes.  1 = serial.")
    args = parser.parse_args()
    main(args.n_per_class, args.workers)
