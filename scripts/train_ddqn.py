"""Train DDQN agents (one per seed) and compare them with the baselines.

  python scripts/train_ddqn.py --synthetic --midday-dip-bps 150 --seeds 3     # sanity check: learnable pattern
  python scripts/train_ddqn.py --data "data/BTCUSDT-1m-*.zip" --seeds 5 --export results.json

Uses CUDA automatically when available. Models are selected on the validation split; the test
split is only evaluated with --reveal-test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import base_parser, config_from_args, load_bars
from rlexec import (BuyAtEnd, BuyAtStart, PerfectForesight, RandomPolicy, TWAP, build, evaluate, load_config,
                    summarize)
from rlexec.ddqn import DDQNConfig, train_ddqn
from rlexec.export import export_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of independently trained agents")
    ap.add_argument("--epochs", type=int, default=None, help="override ddqn.epochs from the config")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--save-dir", default=None, help="save each seed's network here")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.epochs:
        cfg["ddqn"]["epochs"] = args.epochs
    dcfg = DDQNConfig(**cfg["ddqn"])
    bars, label, note = load_bars(args)
    market, costs, env, splits = build(cfg, bars)
    name = "test" if args.reveal_test else "val"
    days = splits[name]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(costs.describe(), "\n")
    print("days: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f"  -> reporting '{name}'")
    print(f"training {args.seeds} seeds x {dcfg.epochs} epochs on {device}; state = {env.obs_names}; "
          f"actions = {env.action}\n")

    agents, runs = [], []
    for seed in range(args.seeds):
        pol, hist = train_ddqn(env, splits["train"], splits["val"], dcfg, seed=seed, device=device)
        pol.name = f"ddqn_s{seed}"
        agents.append(pol)
        runs.append(hist)
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": pol.net.state_dict(), "norm_mean": pol.norm.mean, "norm_std": pol.norm.std,
                        "config": cfg, "obs_names": env.obs_names}, Path(args.save_dir) / f"ddqn_s{seed}.pt")

    baselines = [TWAP(), BuyAtStart(), BuyAtEnd(), RandomPolicy(0), PerfectForesight()]
    res = evaluate(env, days, baselines + agents)
    is_seed = res["policy"].str.startswith("ddqn_s")
    per_seed, base = res[is_seed], res[~is_seed]
    # "ddqn" = average over seeds, day by day: the result you should expect from one training run
    avg = per_seed.groupby("date", as_index=False).mean(numeric_only=True)
    avg.insert(1, "policy", "ddqn")
    combined = pd.concat([base, avg], ignore_index=True)

    twap_mean = base.loc[base.policy == "twap", "is_bps"].mean()
    seed_means = per_seed.groupby("policy")["is_bps"].mean() - twap_mean
    for run, pol in zip(runs, agents):
        run[f"{name}_vs_twap"] = round(float(seed_means[pol.name]), 4)

    pd.set_option("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format)
    print(f"\nShortfall vs arrival price on '{name}', bps (lower is better; vs_twap < 0 beats TWAP):")
    print(summarize(combined))
    v = seed_means.to_numpy()
    print(f"\nPer-seed mean vs TWAP: " + ", ".join(f"{x:+.2f}" for x in v) +
          f"   (spread across seeds: std {v.std(ddof=1) if len(v) > 1 else 0:.2f} bps)")

    if name == "val":
        print("Note: validation was used to pick each agent's best epoch, so the DDQN's validation figure is "
              "optimistic. The test split (--reveal-test) is the unbiased check; use it once, at the end.")
        note = (note + " " if note else "") + (
            "The DDQN's best epoch was chosen on this validation split, so its figure here is optimistic; "
            "the untouched test split is the unbiased check.")

    if args.export:
        median = agents[int(np.argsort(v)[len(v) // 2])]  # schedules shown in the viewer: the median seed
        shown = type(median)(median.net, median.norm, median.device, name="ddqn")
        export_json(args.export, env, cfg, costs, days, baselines + [shown], combined,
                    {"label": args.label or label, "note": note, "synthetic": bool(args.synthetic), "split": name,
                     "split_sizes": {k: len(v2) for k, v2 in splits.items()}},
                    extra={"training": {"device": device, "split": name, "runs": runs}})
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
