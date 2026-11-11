# rl-exec-lab

A test bench for reinforcement-learning execution agents. The task: buy a fixed quantity within
one day at the lowest cost. RL agents (DQN, Double DQN, Dueling Double DQN, PPO) are compared with
simple baselines under a transparent, configurable cost model, with safeguards against fooling
yourself.

**Web page:** https://hugowei.github.io/rl-exec-lab/ (results viewer and run builder, served from `docs/`)

## Use it

1. On the web page, open **New run**, choose the data, state features, methods and cost model,
   and download `run.yaml`.
2. Run it on your computer (uses your GPU when PyTorch sees one):

       git clone https://github.com/HugoWEI/rl-exec-lab && cd rl-exec-lab
       pip install numpy pandas pyyaml torch
       python scripts/train.py --config run.yaml

   This downloads the Binance data it needs, trains every method over several seeds, evaluates,
   and writes `results.json`.
3. Back on the page, click **Open a results file** and choose `results.json`. The file stays in
   your browser.

Without a config, `python scripts/train.py` uses `configs/default.yaml`. Quick checks:

    python -m pytest -q tests
    python scripts/train.py --synthetic --midday-dip-bps 150 --seeds 2 --epochs 5   # agents learn a planted pattern
    python scripts/run_baselines.py --synthetic                                      # baselines only

The same run can be started on GitHub from the Actions tab ("train"); it publishes the results as
the page's default view.

## How it works

- **Episode**: one UTC day, one decision every `step_minutes`. The agent picks how much to buy and
  is filled at the bar open plus costs. Anything left at the last step is bought there.
- **Metric**: implementation shortfall vs the day's first price, in bps (lower is better), and
  the difference vs TWAP with a paired bootstrap confidence interval over days.
- **Costs** (`rlexec/costs.py`): fees, spread, temporary impact, permanent impact and a
  participation cap, each with its own switch. Every result is split into price timing plus each
  cost component, and the active formulas are printed with their parameters.
- **Baselines**: TWAP, buy at start, buy at end, random, and perfect foresight under the same cost
  model (an exact bound when permanent impact is off).
- **State features** (`rlexec/data.py`, `FEATURES`): returns over several horizons, volatility,
  relative volume, taker buy/sell imbalance, position in the day's range, deviation from the day's
  VWAP, time of day. Time remaining and inventory remaining are always included.
- **Agents** (`rlexec/agents.py`): DQN, Double DQN, Dueling Double DQN and PPO, all small MLPs.
  Actions are multiples of the TWAP rate (1 = TWAP). Training uses a reward measured relative to
  TWAP's inventory path, which has the same optimum as the shortfall with much less noise.
  Each method is trained over several seeds; its row is the day-by-day average over seeds.
- **Bias controls**: features use past bars only (tested); chronological train/validation/test
  split with an embargo; validation reported by default and test only with `--reveal-test`;
  several seeds per method; an overfitting check that scores every agent on its training days
  next to the held-out days, per epoch and at the end; the final-epoch result is shown next to the best-epoch result, because
  picking the best epoch on validation makes that figure optimistic.

## Layout

    rlexec/costs.py      cost model
    rlexec/data.py       download, loading, synthetic bars, feature registry, day slicing, splits
    rlexec/env.py        execution environment
    rlexec/baselines.py  reference policies and the hindsight-optimal schedule
    rlexec/agents.py     RL agents and training loops
    rlexec/evaluate.py   per-day results and bootstrap summary
    rlexec/export.py     results file for the web page
    rlexec/config.py     config loading and command-line options
    scripts/             train.py, run_baselines.py, download_binance.py, export_schema.py
    docs/                the web page (GitHub Pages)

After adding a feature or a method, run `python scripts/export_schema.py` so the page's run builder
lists it.

## Known limits

- Impact parameters are assumptions; calibrating them needs real fills.
- Fills are at bar opens; there is no order book or queue model.
- One chronological train/validation/test split; no rolling walk-forward folds yet.
- Findings so far on BTCUSDT 5-minute bars: no agent has shown an edge over TWAP that survives
  the checks above (see `results/`).
