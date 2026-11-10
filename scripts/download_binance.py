"""Download monthly 1-minute klines from Binance's public data site.

  python scripts/download_binance.py --symbol BTCUSDT --start 2023-01 --end 2024-12
"""
import argparse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{ym}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for month in pd.period_range(args.start, args.end, freq="M"):
        url = BASE.format(s=args.symbol, ym=str(month))
        dest = out / url.rsplit("/", 1)[1]
        if dest.exists():
            print("have", dest.name)
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print("ok  ", dest.name)
        except Exception as e:  # missing month or network error: report and continue
            print("FAIL", dest.name, e)


if __name__ == "__main__":
    main()
