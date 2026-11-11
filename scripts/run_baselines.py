"""Evaluate the baseline policies (no training).

  python scripts/run_baselines.py --synthetic
  python scripts/run_baselines.py --symbol BTCUSDT --start 2023-01 --end 2024-12
  python scripts/run_baselines.py --config run.yaml --reveal-test   # use the test split sparingly
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec import (BuyAtEnd, BuyAtStart, PerfectForesight, RandomPolicy, TWAP, add_common_args, build,  # noqa: E402
                    config_from_args, evaluate, resolve_bars, summarize)
from rlexec.export import export_json  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--export", help="write a results JSON for the web viewer")
    args = ap.parse_args()
    cfg = config_from_args(args)
    bars, label, note = resolve_bars(cfg)
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
    if args.export:
        export_json(args.export, env, cfg, costs, days, policies, results,
                    {"label": args.label or label, "note": note, "synthetic": cfg["data"]["source"] == "synthetic",
                     "split": name, "split_sizes": {k: len(v) for k, v in splits.items()}})
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
