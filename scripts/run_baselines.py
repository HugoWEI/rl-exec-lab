"""Evaluate the baseline policies.

  python scripts/run_baselines.py --synthetic
  python scripts/run_baselines.py --data "data/BTCUSDT-1m-*.zip"
  python scripts/run_baselines.py --data "data/*.zip" --reveal-test   # use sparingly
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec.export import export_json
from rlexec import (BuyAtEnd, BuyAtStart, CostModel, ExecutionEnv, Market, PerfectForesight, RandomPolicy,
                    TWAP, chronological_split, evaluate, load_binance_klines, summarize, synthetic_bars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/default.yaml"))
    ap.add_argument("--data", help="glob of Binance kline csv/zip files")
    ap.add_argument("--synthetic", action="store_true", help="use driftless synthetic bars")
    ap.add_argument("--reveal-test", action="store_true", help="report the test split instead of validation")
    ap.add_argument("--out", help="write per-day results to this csv")
    ap.add_argument("--export", help="write a results JSON for the web viewer (docs/index.html)")
    ap.add_argument("--label", default=None, help="dataset label shown in the viewer")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.synthetic:
        bars = synthetic_bars()
    elif args.data:
        files = glob.glob(args.data)
        if not files:
            sys.exit(f"no files match {args.data}")
        bars = load_binance_klines(files)
    else:
        sys.exit("pass --data or --synthetic")

    market = Market(bars, **cfg["market"])
    costs = CostModel.from_dict(cfg["costs"])
    env = ExecutionEnv(market, costs, cfg["parent"], cfg["action_grid"])
    splits = chronological_split(market.days, **cfg["split"])
    name = "test" if args.reveal_test else "val"
    days = splits[name]

    print(costs.describe(), "\n")
    print("days: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f"  -> reporting '{name}'")
    if not days:
        sys.exit("not enough data for this split")
    policies = [TWAP(), BuyAtStart(), BuyAtEnd(), RandomPolicy(0), PerfectForesight()]
    results = evaluate(env, days, policies)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format)
    print("\nShortfall vs arrival price, bps (lower is better; vs_twap < 0 beats TWAP):")
    print(summarize(results))
    if args.out:
        results.to_csv(args.out, index=False)
    if args.export:
        label = args.label or ("Synthetic driftless bars (demo)" if args.synthetic else args.data)
        note = ("Synthetic, driftless prices: nothing is predictable, so no policy should beat TWAP "
                "except perfect foresight." if args.synthetic else "")
        export_json(args.export, env, cfg, costs, days, policies, results,
                    {"label": label, "note": note, "synthetic": bool(args.synthetic), "split": name,
                     "split_sizes": {k: len(v) for k, v in splits.items()}})
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
