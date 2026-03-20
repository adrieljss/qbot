"""
download_data.py
----------------
Downloads 1-hour OHLCV kline data from Binance Data Vision (public, no API key needed).
Saves one CSV per symbol into ./data/  with columns:
  datetime, open, high, low, close, volume, quote_volume

Binance Data Vision layout for hourly data:
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY}-{MM}.zip

Usage:
    python download_data.py                          # default symbols & date range
    python download_data.py --start 2022-01-01 --end 2024-12-31
    python download_data.py --symbols BTC ETH SOL    # USDT pairs assumed
"""

import os
import io
import zipfile
import argparse
import requests
import pandas as pd
from datetime import date

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "TRXUSDT", "SUIUSDT", "ADAUSDT", "LINKUSDT"
]

DEFAULT_START = "2022-01-01"
DEFAULT_END   = "2024-12-31"

BASE_URL  = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL  = "1h"
DATA_DIR  = "./data"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_range(start: str, end: str):
    """Yield (year, month) tuples covering start through end inclusive."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    year, month = s.year, s.month
    while (year, month) <= (e.year, e.month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def fetch_month(session: requests.Session, symbol: str,
                year: int, month: int) -> pd.DataFrame | None:
    """
    Fetch one month of 1h klines from Binance Data Vision.
    Returns a DataFrame (one row per hour) or None if the file doesn't exist yet.
    """
    fname = f"{symbol}-{INTERVAL}-{year}-{month:02d}.zip"
    url   = f"{BASE_URL}/{symbol}/{INTERVAL}/{fname}"
    resp  = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)
    return df


def download_symbol(symbol: str, start: str, end: str,
                    session: requests.Session) -> pd.DataFrame:
    """Download and concatenate all monthly 1h files for one symbol."""
    try:
        from tqdm import tqdm
        months = list(month_range(start, end))
        iterator = tqdm(months, desc=symbol, leave=False, ncols=80)
    except ImportError:
        months = list(month_range(start, end))
        iterator = months

    frames = []
    for year, month in iterator:
        try:
            df = fetch_month(session, symbol, year, month)
            if df is not None:
                frames.append(df)
        except Exception as e:
            print(f"  Warning: {symbol} {year}-{month:02d} failed — {e}")

    if not frames:
        print(f"  No data fetched for {symbol}")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Parse timestamps — open_time is milliseconds epoch
    out["datetime"]     = (pd.to_datetime(out["open_time"], unit="us")
                             .dt.strftime("%Y-%m-%d %H:%M:%S"))
    out["open"]         = out["open"].astype(float)
    out["high"]         = out["high"].astype(float)
    out["low"]          = out["low"].astype(float)
    out["close"]        = out["close"].astype(float)
    out["volume"]       = out["volume"].astype(float)
    out["quote_volume"] = out["quote_volume"].astype(float)

    # Trim to exact requested date range
    out["_dt"] = pd.to_datetime(out["datetime"])
    start_ts   = pd.Timestamp(start)
    end_ts     = pd.Timestamp(end) + pd.Timedelta(days=1)
    out = out[(out["_dt"] >= start_ts) & (out["_dt"] < end_ts)]

    out = (out[["datetime", "open", "high", "low", "close", "volume", "quote_volume"]]
             .sort_values("datetime")
             .drop_duplicates("datetime")
             .reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Binance 1h klines")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Base symbols e.g. BTC ETH SOL (USDT pairs assumed)")
    parser.add_argument("--start",  default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=DEFAULT_END,   help="End date   YYYY-MM-DD")
    parser.add_argument("--outdir", default=DATA_DIR,      help="Output directory")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.upper() + ("USDT" if not s.upper().endswith("USDT") else "")
                   for s in args.symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Downloading {len(symbols)} symbols @ 1h  |  {args.start} → {args.end}")
    print(f"Output dir : {os.path.abspath(args.outdir)}\n")

    session = requests.Session()
    session.headers.update({"User-Agent": "binance-hourly-downloader/1.0"})

    for symbol in symbols:
        out_path = os.path.join(args.outdir, f"{symbol}.csv")
        print(f"[{symbol}]")
        df = download_symbol(symbol, args.start, args.end, session)
        if df.empty:
            continue
        df.to_csv(out_path, index=False)
        n_days = df["datetime"].str[:10].nunique()
        print(f"  Saved {len(df):,} rows ({n_days} days) → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()