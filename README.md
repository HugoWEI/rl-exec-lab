# rl-exec-lab

Test bench for execution policies: buy a fixed quantity within one day at the lowest cost.
Current state: environment, configurable cost model, baselines, evaluation and a web results viewer. No RL yet.

**Viewer:** https://hugowei.github.io/rl-exec-lab/ (GitHub Pages, served from `docs/`)

## Quick start

    pip install numpy pandas pyyaml pytest
    python -m pytest -q tests
    python scripts/run_baselines.py --synthetic

    # real data (Binance public 1-minute klines)
    python scripts/download_binance.py --symbol BTCUSDT --start 2023-01 --end 2024-12
    python scripts/run_baselines.py --data "data/BTCUSDT-1m-*.zip"

## Viewing results

    python scripts/run_baselines.py --data "data/BTCUSDT-1m-*.zip" --export results.json

Open the viewer and use "Open a results file" to load `results.json` (it stays in your browser).
To change the demo shown by default, export to `docs/data/demo.json` and push.

## How it works

- **Episode**: one UTC day, one decision every `step_minutes`. The agent picks a fraction of its
  remaining inventory and is filled at the bar open plus costs. Anything left at the last step is
  bought there (forced completion).
- **Metric**: implementation shortfall vs the day's arrival price, in bps, and the difference vs
  TWAP with a paired bootstrap CI over days. The sum of step rewards equals minus the shortfall.
- **Costs** (`configs/default.yaml`): fees, spread, temporary impact, permanent impact and a
  participation cap, each with its own on/off switch. `CostModel.describe()` prints the active
  formulas with the current parameters; every result is split into timing + each cost component.
- **Baselines**: buy at start, buy at end, TWAP, random, and perfect foresight under the same
  cost model (exact bound when permanent impact is off).
- **Bias controls**: features use past bars only (tested), chronological train/val/test split with
  an embargo, validation reported by default, test only with `--reveal-test`.

## Layout

    rlexec/costs.py      cost model
    rlexec/data.py       loading, synthetic bars, feature registry, day slicing, splits
    rlexec/env.py        execution environment
    rlexec/baselines.py  reference policies and the hindsight-optimal schedule
    rlexec/evaluate.py   per-day results and bootstrap summary
    rlexec/export.py     results JSON for the viewer
    docs/                static viewer (GitHub Pages)

## Known limits

- Impact parameters are assumptions; calibrating them needs real fills.
- Fills are at bar opens; there is no order book or queue model.
- Features are not normalised yet (to be fitted on the training split in step 2).
