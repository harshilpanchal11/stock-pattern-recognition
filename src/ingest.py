"""
Stage 1 — Data Ingestion
Pulls daily OHLCV data for S&P 500 stocks via yfinance.
Saves each ticker as a Parquet file in data/raw/.

Run: python src/ingest.py
"""

import yfinance as yf
import pandas as pd
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────
TICKERS = [
    # Technology
    'AAPL','MSFT','NVDA','GOOGL','META','AMZN','TSLA','AMD','INTC','CRM',
    # Financials
    'JPM','BAC','GS','MS','WFC','BLK','AXP','V','MA','C',
    # Healthcare
    'JNJ','UNH','PFE','ABBV','MRK','LLY','TMO','ABT','DHR','BMY',
    # Energy
    'XOM','CVX','COP','SLB','EOG','MPC','VLO','OXY','HAL','PXD',
    # Consumer
    'WMT','PG','KO','PEP','COST','MCD','NKE','SBUX','TGT','HD',
    # Industrials
    'BA','CAT','GE','HON','UPS','LMT','RTX','DE','MMM','EMR',
    # Communication / Media
    'DIS','NFLX','CMCSA','T','VZ',
    # Utilities / Materials
    'NEE','DUK','LIN','APD','NEM',
]

START_DATE = "2020-01-01"
END_DATE   = "2026-03-31"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────
def download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker, start=START_DATE, end=END_DATE,
            interval="1d", auto_adjust=True, progress=False
        )
        if df.empty or len(df) < 90:
            print(f"  SKIP {ticker}: only {len(df)} rows")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        return None


def validate(df: pd.DataFrame, ticker: str) -> bool:
    # Check for non-positive prices
    if (df[['Open','High','Low','Close']] <= 0).any().any():
        print(f"  WARNING {ticker}: non-positive prices found — skipping")
        return False
    # Check for large date gaps (> 10 calendar days between consecutive rows)
    gaps = df.index.to_series().diff().dt.days
    max_gap = gaps.max()
    if max_gap > 10:
        print(f"  WARNING {ticker}: max gap = {max_gap} days (possible missing data)")
    return True


# ── Main ─────────────────────────────────────────────────────────────
def main():
    saved, skipped, errors = 0, 0, 0

    for ticker in TICKERS:
        out_path = OUTPUT_DIR / f"{ticker}.parquet"

        if out_path.exists():
            print(f"  CACHED  {ticker}")
            saved += 1
            continue

        print(f"Downloading {ticker}...")
        df = download_ticker(ticker)

        if df is not None and validate(df, ticker):
            df.to_parquet(out_path)
            print(f"  SAVED   {ticker}: {len(df)} rows  "
                  f"({df.index[0].date()} → {df.index[-1].date()})")
            saved += 1
        else:
            skipped += 1

        time.sleep(0.4)   # throttle — avoids Yahoo Finance rate limiting

    print(f"\n── Ingestion complete ──────────────────────────")
    print(f"  Saved:   {saved} tickers")
    print(f"  Skipped: {skipped} tickers")
    print(f"  Output:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
