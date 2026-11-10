import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec import load_binance_klines, synthetic_bars  # noqa: E402


def base_parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="yaml config (default: configs/default.yaml)")
    ap.add_argument("--data", help="glob of Binance kline csv/zip files")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic bars")
    ap.add_argument("--synthetic-days", type=int, default=120)
    ap.add_argument("--midday-dip-bps", type=float, default=0.0,
                    help="synthetic only: add a learnable average midday price dip of this size")
    ap.add_argument("--reveal-test", action="store_true", help="report the test split instead of validation")
    ap.add_argument("--export", help="write a results JSON for the web viewer (docs/index.html)")
    ap.add_argument("--label", default=None, help="dataset label shown in the viewer")
    return ap


def load_bars(args):
    """Returns (bars, label, note)."""
    if args.synthetic:
        bars = synthetic_bars(n_days=args.synthetic_days, midday_dip_bps=args.midday_dip_bps)
        if args.midday_dip_bps:
            return bars, f"Synthetic bars with a {args.midday_dip_bps:g} bps average midday dip (demo)", (
                "Synthetic prices with a built-in average midday dip: a learnable pattern, used to check "
                "that an agent can learn. Not evidence about real markets.")
        return bars, "Synthetic driftless bars (demo)", (
            "Synthetic, driftless prices: nothing is predictable, so no policy should beat TWAP "
            "except perfect foresight.")
    if args.data:
        files = glob.glob(args.data)
        if not files:
            sys.exit(f"no files match {args.data}")
        return load_binance_klines(files), args.data, ""
    sys.exit("pass --data or --synthetic")
