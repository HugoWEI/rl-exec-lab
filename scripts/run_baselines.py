"""Evaluate the baseline policies.

  python scripts/run_baselines.py --synthetic
  python scripts/run_baselines.py --data "data/BTCUSDT-1m-*.zip"
  python scripts/run_baselines.py --data "data/*.zip" --reveal-test   # use sparingly
"""
import sys

import pandas as pd

from _common import base_parser, load_bars
from rlexec import (BuyAtEnd, BuyAtStart, PerfectForesight, RandomPolicy, TWAP, build, evaluate, load_config,
                    summarize)
from rlexec.export import export_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--out", help="write per-day results to this csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bars, label, note = load_bars(args)
    market, costs, env, splits = build(cfg, bars)
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
        export_json(args.export, env, cfg, costs, days, policies, results,
                    {"label": args.label or label, "note": note, "synthetic": bool(args.synthetic), "split": name,
                     "split_sizes": {k: len(v) for k, v in splits.items()}})
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
