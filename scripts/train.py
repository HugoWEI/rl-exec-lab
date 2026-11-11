"""Train RL agents and compare them with the baselines. One command does everything:
downloads the data it needs, trains each method over several seeds, evaluates, and writes
results.json for the web viewer.

  python scripts/train.py --config run.yaml                 # a config built on the web page
  python scripts/train.py                                    # defaults: BTCUSDT 2023-2024, all methods
  python scripts/train.py --synthetic --midday-dip-bps 150 --methods ddqn,ppo --seeds 2 --epochs 6

Uses CUDA automatically when available. Models are selected on the validation split; the test
split is only evaluated with --reveal-test.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec import (BuyAtEnd, BuyAtStart, PerfectForesight, RandomPolicy, TWAP, add_common_args, build,  # noqa: E402
                    config_from_args, evaluate, resolve_bars, summarize)
from rlexec.agents import METHODS, train_agent  # noqa: E402
from rlexec.export import export_json  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--methods", default=None, help="comma-separated, from: " + ", ".join(METHODS))
    ap.add_argument("--seeds", type=int, default=None, help="independently trained agents per method")
    ap.add_argument("--epochs", type=int, default=None, help="override the number of training epochs")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--save-dir", default=None, help="save each trained network here")
    ap.add_argument("--export", default="results.json", help="results file for the web viewer")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.methods:
        cfg["methods"] = args.methods.split(",")
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.epochs:
        cfg["dqn"]["epochs"] = cfg["ppo"]["epochs"] = args.epochs
    unknown = [m for m in cfg["methods"] if m not in METHODS]
    if unknown:
        sys.exit(f"unknown methods {unknown}; available: {', '.join(METHODS)}")

    bars, label, note = resolve_bars(cfg)
    market, costs, env, splits = build(cfg, bars)
    name = "test" if args.reveal_test else "val"
    days = splits[name]
    if not splits["train"] or not splits["val"] or not days:
        sys.exit("not enough data for the train/validation/test split")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(costs.describe(), "\n")
    print("days: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f"  -> reporting '{name}'")
    print(f"methods {cfg['methods']} x {cfg['seeds']} seeds on {device}\nstate = {env.obs_names}\nactions = {env.action}\n")

    agents, runs = {}, []
    for method in cfg["methods"]:
        for seed in range(cfg["seeds"]):
            pol, hist = train_agent(method, env, splits["train"], splits["val"], cfg, seed, device)
            agents[(method, seed)] = pol.renamed(f"{method}__s{seed}")
            runs.append(hist)
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": pol.net.state_dict(), "norm_mean": pol.norm.mean, "norm_std": pol.norm.std,
                            "config": cfg, "obs_names": env.obs_names, "method": method},
                           Path(args.save_dir) / f"{method}_s{seed}.pt")

    baselines = [TWAP(), BuyAtStart(), BuyAtEnd(), RandomPolicy(0), PerfectForesight()]
    res = evaluate(env, days, baselines + list(agents.values()))
    is_agent = res["policy"].str.contains("__s")
    per_seed, combined = res[is_agent].copy(), res[~is_agent]
    twap_mean = combined.loc[combined.policy == "twap", "is_bps"].mean()
    seed_means = per_seed.groupby("policy")["is_bps"].mean() - twap_mean
    for run in runs:
        run[f"{name}_vs_twap"] = round(float(seed_means[f"{run['method']}__s{run['seed']}"]), 4)

    # overfitting check: the same selected agents, scored on all the days they were trained on
    if agents:
        tr = evaluate(env, splits["train"], [TWAP()] + list(agents.values()))
        tr_means = tr.groupby("policy")["is_bps"].mean()
        for run in runs:
            run["train_vs_twap"] = round(float(tr_means[f"{run['method']}__s{run['seed']}"] - tr_means["twap"]), 4)

    # one row per method = average over its seeds, day by day: what to expect from one training run
    per_seed["method"] = per_seed["policy"].str.split("__s").str[0]
    shown = []
    for method in cfg["methods"]:
        avg = per_seed[per_seed.method == method].groupby("date", as_index=False).mean(numeric_only=True)
        avg.insert(1, "policy", method)
        combined = pd.concat([combined, avg], ignore_index=True)
        scores = [seed_means[f"{method}__s{s}"] for s in range(cfg["seeds"])]
        median_seed = int(np.argsort(scores)[len(scores) // 2])  # schedules shown in the viewer: the median seed
        shown.append(agents[(method, median_seed)].renamed(method))

    pd.set_option("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format)
    print(f"\nShortfall vs arrival price on '{name}', bps (lower is better; vs_twap < 0 beats TWAP):")
    print(summarize(combined))
    print(f"\nPer-seed mean vs TWAP on '{name}' (and at the final epoch on validation, which involves no selection):")
    for method in cfg["methods"]:
        rs = [r for r in runs if r["method"] == method]
        v = np.array([r[f"{name}_vs_twap"] for r in rs])
        final = np.array([r["final_val_vs_twap"] for r in rs])
        print(f"  {METHODS[method]['label']:<20} " + ", ".join(f"{x:+.2f}" for x in v) +
              f"   | final epoch: mean {final.mean():+.2f}")
    if agents:
        print(f"\nOverfitting check, mean of seeds, bps vs TWAP (a large gap = the agents memorised training days):")
        print(f"  {'':<20} {'training days':>14} {name + ' days':>14} {'gap':>8}")
        for method in cfg["methods"]:
            rs = [r for r in runs if r["method"] == method]
            a, b = np.mean([r["train_vs_twap"] for r in rs]), np.mean([r[f"{name}_vs_twap"] for r in rs])
            print(f"  {METHODS[method]['label']:<20} {a:>+14.2f} {b:>+14.2f} {b - a:>+8.2f}")
    if name == "val" and cfg["methods"]:
        print("Note: validation was used to pick each agent's best epoch, so the agents' validation figures are "
              "optimistic. The test split (--reveal-test) is the unbiased check; use it once, at the end.")
        note = (note + " " if note else "") + (
            "Each agent's best epoch was chosen on this validation split, so the agents' figures here are "
            "optimistic; the untouched test split is the unbiased check.")

    export_json(args.export, env, cfg, costs, days, baselines + shown, combined,
                {"label": args.label or label, "note": note, "synthetic": cfg["data"]["source"] == "synthetic",
                 "split": name, "split_sizes": {k: len(v) for k, v in splits.items()}},
                extra={"training": {"device": device, "split": name, "runs": runs,
                                    "methods": [{"name": m, "label": METHODS[m]["label"]} for m in cfg["methods"]]}})
    print(f"\nwrote {args.export}  (open it in the viewer with 'Open a results file')")


if __name__ == "__main__":
    main()
