"""Write a results JSON for the static web viewer (docs/index.html)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .env import run_episode
from .evaluate import summarize


def export_json(path, env, cfg: dict, costs, days: list, policies: list, results, meta: dict,
                n_sample_days: int = 6) -> None:
    summ = summarize(results).reset_index()
    wide = results.pivot(index="date", columns="policy", values="is_bps")
    pick = np.unique(np.linspace(0, len(days) - 1, min(n_sample_days, len(days))).astype(int))
    samples = []
    for i in pick:
        day = days[i]
        sched = {}
        for pol in policies:
            r = run_episode(env, day, pol)
            sched[pol.name] = np.round(100 * np.cumsum(r.fills) / r.fills.sum(), 2).tolist()
        samples.append({"date": str(day.date.date()), "price": np.round(day.open, 4).tolist(), "bought_pct": sched})
    out = {
        "meta": {**meta, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_days": len(days), "first_day": str(days[0].date.date()), "last_day": str(days[-1].date.date())},
        "config": {**cfg, "costs": costs.to_dict()},
        "cost_description": costs.describe(),
        "policies": [{"name": p.name, "uses_future": p.uses_future} for p in policies],
        "summary": json.loads(summ.round(4).to_json(orient="records")),
        "daily": {"dates": [str(d.date()) for d in wide.index],
                  "shortfall_bps": {c: wide[c].round(3).tolist() for c in wide.columns}},
        "samples": samples,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, separators=(",", ":")))
