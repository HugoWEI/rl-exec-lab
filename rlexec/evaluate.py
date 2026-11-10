"""Run policies over a set of days and summarise with paired bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import COMPONENTS
from .env import ExecutionEnv, run_episode

BREAKDOWN = ["timing", *COMPONENTS]


def evaluate(env: ExecutionEnv, days: list, policies: list) -> pd.DataFrame:
    """One row per (day, policy): shortfall in bps and its breakdown."""
    rows = []
    for day in days:
        for pol in policies:
            r = run_episode(env, day, pol)
            rows.append({"date": day.date, "policy": pol.name, "is_bps": r.is_bps,
                         **r.breakdown_bps, "forced_pct": r.forced_pct})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, benchmark: str = "twap", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean shortfall per policy, plus mean difference vs the benchmark with a 95% CI from a
    paired bootstrap over days. Negative `vs_<benchmark>` means cheaper than the benchmark."""
    wide = results.pivot(index="date", columns="policy", values="is_bps")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    means = results.groupby("policy")[["is_bps", *BREAKDOWN, "forced_pct"]].mean()
    out = means.rename(columns={"is_bps": "shortfall_bps"})
    diff = wide.sub(wide[benchmark], axis=0)
    boot = diff.to_numpy()[idx].mean(axis=1)  # (n_boot, n_policies)
    out[f"vs_{benchmark}"] = diff.mean()
    out["ci_low"] = pd.Series(np.percentile(boot, 2.5, axis=0), index=diff.columns)
    out["ci_high"] = pd.Series(np.percentile(boot, 97.5, axis=0), index=diff.columns)
    out["n_days"] = len(wide)
    cols = ["shortfall_bps", f"vs_{benchmark}", "ci_low", "ci_high", *BREAKDOWN, "forced_pct", "n_days"]
    return out[cols].sort_values("shortfall_bps")
