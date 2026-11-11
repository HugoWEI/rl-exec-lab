"""Look at the state features themselves, on the TRAINING split only (so nothing leaks into
validation or test): their distributions, how correlated they are with each other, and whether any
of them says anything about where the price goes next.

  python scripts/analyze_features.py --synthetic --midday-dip-bps 150
  python scripts/analyze_features.py --symbol BTCUSDT --start 2023-01 --end 2024-12 --export docs/data/features.json

Predictive power is the rank correlation, pooled over all training days, between a feature and what
the price does afterwards. Its t-statistic treats whole days as the independent units.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlexec import FEATURES, add_common_args, build, config_from_args, resolve_bars  # noqa: E402

TARGETS = [("next_1", "next step"), ("next_12", "next 12 steps"), ("next_48", "next 48 steps"),
           ("rest_of_day", "rest of day (average later price vs now)")]


def forward_targets(open_: np.ndarray) -> dict:
    """Log price changes after each step; NaN where the day ends first."""
    n, lp, out = len(open_), np.log(open_), {}
    for k in (1, 12, 48):
        v = np.full(n, np.nan)
        v[:-k] = lp[k:] - lp[:-k]
        out[f"next_{k}"] = v
    later_mean = np.full(n, np.nan)  # what waiting earns: the average of all later prices vs the current one
    csum = np.cumsum(open_[::-1])[::-1]
    later_mean[:-1] = csum[1:] / np.arange(n - 1, 0, -1)
    out["rest_of_day"] = np.log(later_mean / open_)
    return out


def _ranks(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank().to_numpy()


def _cluster(per_day: np.ndarray, total_weight: float):
    """Estimate = sum of per-day contributions / total weight; t-statistic treats whole days as the
    independent units (observations inside a day overlap and are strongly dependent)."""
    d = len(per_day)
    est = per_day.sum(0) / total_weight
    se = np.sqrt(d * per_day.var(0, ddof=1)) / total_weight
    return est, np.where(se > 0, est / np.where(se > 0, se, 1), np.nan)


def predictive_power(days: list):
    """Pooled rank correlation of each feature with each later price move, and the gap in the
    rest-of-day move between a feature's top and bottom 20% of values.

    Everything is pooled across days with no within-day centring. (Centring inside each day would
    make any price-level feature look predictive even on a pure random walk, because a day's high
    points are by construction followed by lower ones within that same day.)"""
    X = np.concatenate([d.features for d in days])
    day = np.concatenate([np.full(d.n, i) for i, d in enumerate(days)])
    tg = [forward_targets(d.open) for d in days]
    D, half = len(days), len(days) // 2
    onehot = lambda idx: np.eye(D)[idx].T  # (D, n) matrix summing rows into days
    ic = {}
    for t, _ in TARGETS:
        y = np.concatenate([g[t] for g in tg])
        ok = ~np.isnan(y)
        rx, ry = _ranks(X[ok]), _ranks(y[ok][:, None])[:, 0]
        sx = rx.std(0)
        zx = (rx - rx.mean(0)) / np.where(sx > 0, sx, 1)
        zy = (ry - ry.mean()) / ry.std()
        per_day = onehot(day[ok]) @ (zx * zy[:, None])
        n_day = np.bincount(day[ok], minlength=D)
        est, tstat = _cluster(per_day, ok.sum())
        ic[t] = {"ic": est, "t": tstat, "first_half": per_day[:half].sum(0) / max(1, n_day[:half].sum()),
                 "second_half": per_day[half:].sum(0) / max(1, n_day[half:].sum())}
    y = np.concatenate([g["rest_of_day"] for g in tg])
    ok = ~np.isnan(y)
    Xo, yo, M = X[ok], y[ok] * 1e4, onehot(day[ok])
    lo, hi = np.percentile(Xo, 20, axis=0), np.percentile(Xo, 80, axis=0)
    top, bot = Xo >= hi, Xo <= lo
    w = top / np.maximum(top.sum(0), 1) - bot / np.maximum(bot.sum(0), 1)
    spread, spread_t = _cluster(M @ (w * yo[:, None]), 1.0)
    return ic, spread, spread_t


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--export", help="write a JSON for the web page's Features tab")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if not args.features:
        cfg["market"]["features"] = list(FEATURES)  # analyse everything unless told otherwise
    bars, label, _ = resolve_bars(cfg)
    market, _, _, splits = build(cfg, bars)
    days, names = splits["train"], market.feature_names
    if len(days) < 6:
        sys.exit("not enough training days")
    X = np.concatenate([d.features for d in days])
    n_tests = len(names) * len(TARGETS)
    # two-sided 5% threshold after correcting for the number of feature x horizon tests (Bonferroni)
    from statistics import NormalDist
    t_crit = NormalDist().inv_cdf(1 - 0.025 / n_tests)

    ic, spread, spread_t = predictive_power(days)
    rows, out_feats, out_ic, out_spread = [], [], {}, {}
    for j, name in enumerate(names):
        x = X[:, j]
        m, sd = x.mean(), x.std()
        z = (x - m) / sd if sd > 0 else x * 0
        lo, hi = np.percentile(x, [0.5, 99.5])
        counts, _ = np.histogram(np.clip(x, lo, hi), bins=40, range=(lo, hi) if hi > lo else (lo - 1, lo + 1))
        out_feats.append({"name": name, "description": FEATURES[name][0], "mean": float(m), "std": float(sd),
                          "skew": float((z**3).mean()), "excess_kurtosis": float((z**4).mean() - 3),
                          "p1": float(np.percentile(x, 1)), "p50": float(np.median(x)), "p99": float(np.percentile(x, 99)),
                          "hist": {"lo": float(lo), "hi": float(hi), "counts": counts.tolist()}})
        out_ic[name], row = {}, {"feature": name}
        for t, _ in TARGETS:
            mu, tt = float(ic[t]["ic"][j]), float(ic[t]["t"][j])
            out_ic[name][t] = {"ic": mu, "t": tt, "first_half": float(ic[t]["first_half"][j]),
                               "second_half": float(ic[t]["second_half"][j])}
            row[t] = f"{100 * mu:+6.2f} (t {tt:+5.1f}){'*' if abs(tt) > t_crit else ' '}"
        mu, tt = float(spread[j]), float(spread_t[j])
        out_spread[name] = {"bps": mu, "t": tt}
        row["top-bottom 20%, bps"] = f"{mu:+6.2f} (t {tt:+5.1f})"
        rows.append(row)

    sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 200_000), replace=False)]
    corr = np.nan_to_num(np.corrcoef(_ranks(sub), rowvar=False))

    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 100)
    print(f"{label}\nTraining split only: {len(days)} days ({days[0].date.date()} to {days[-1].date.date()}), "
          f"{cfg['market']['step_minutes']}-minute steps, {len(X):,} observations\n")
    print("Distributions:")
    print(pd.DataFrame(out_feats).drop(columns=["description", "hist"]).set_index("name").round(4))
    print(f"\nPredictive power: rank correlation x 100 with the later price move (t-statistic, days as units). "
          f"* = |t| > {t_crit:.2f}, the 5% bar after correcting for {n_tests} tests.")
    print(pd.DataFrame(rows).set_index("feature").to_string())
    pairs = [(abs(corr[i, j]), names[i], names[j], corr[i, j]) for i in range(len(names)) for j in range(i)]
    print("\nMost correlated feature pairs (rank correlation):")
    for _, a, b, c in sorted(pairs, reverse=True)[:8]:
        print(f"  {a:<18} {b:<18} {c:+.2f}")

    if args.export:
        clean = lambda o: json.loads(json.dumps(o).replace("NaN", "null"))
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export).write_text(json.dumps(clean({
            "meta": {"label": args.label or label, "split": "train", "n_days": len(days),
                     "first_day": str(days[0].date.date()), "last_day": str(days[-1].date.date()),
                     "step_minutes": cfg["market"]["step_minutes"], "n_obs": int(len(X)), "t_crit": t_crit,
                     "n_tests": n_tests, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "targets": [{"key": k, "label": l} for k, l in TARGETS],
            "features": out_feats, "ic": out_ic, "spread": out_spread,
            "corr": {"names": names, "matrix": np.round(corr, 3).tolist()}}), separators=(",", ":")))
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
