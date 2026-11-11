"""Download monthly 1-minute klines from Binance's public data site.

  python scripts/download_binance.py --symbol BTCUSDT --start 2023-01 --end 2024-12

(train.py and run_baselines.py download what they need automatically; this is only for fetching data ahead of time.)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec import download_binance  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    download_binance(a.symbol, a.start, a.end, a.out)
