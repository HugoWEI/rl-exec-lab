"""Execution environment: buy a fixed parent quantity within one day.

Timeline of a step t: the agent sees features built from bars before t and the open of bar t,
chooses a fraction of its remaining inventory, and is filled at that open plus costs.
Whatever is left at the final step is bought there (forced completion).

Reward per step = -(cost of this fill relative to the arrival price), in bps of the parent
notional, so the sum of rewards over a day is exactly -implementation shortfall (bps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import COMPONENTS, CostModel
from .data import Day, Market

PRIVATE_FEATURES = ["time_remaining", "inventory_remaining"]  # always in the state (needed for Markov property)


@dataclass
class EpisodeResult:
    date: object
    is_bps: float                 # implementation shortfall vs arrival price, all-in
    breakdown_bps: dict           # timing + each cost component; sums to is_bps
    forced_pct: float             # share of the parent bought above the cap at the final step
    fills: np.ndarray             # quantity bought at each step


class ExecutionEnv:
    def __init__(self, market: Market, costs: CostModel, parent: dict | None = None,
                 action_grid=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0)):
        self.market, self.costs = market, costs
        self.parent = parent or {"mode": "pct_prev_day_volume", "value": 0.01}
        self.action_grid = tuple(action_grid)  # for discrete agents; step() accepts any fraction in [0, 1]
        self.obs_names = market.feature_names + PRIVATE_FEATURES
        self.day: Day | None = None

    def parent_qty(self, day: Day) -> float:
        mode, value = self.parent["mode"], self.parent["value"]
        if mode == "pct_prev_day_volume":
            return value * day.prev_volume
        if mode == "fixed_qty":
            return float(value)
        raise ValueError(f"unknown parent mode {mode!r}")

    def reset(self, day: Day) -> np.ndarray:
        self.day, self.t = day, 0
        self.Q = self.parent_qty(day)
        self.remaining = self.Q
        self.fills = np.zeros(day.n)
        self.acc = {k: 0.0 for k in ("timing",) + COMPONENTS}
        self.forced = 0.0
        return self._obs()

    def cap(self) -> float:
        return float(self.costs.cap_qty(self.day.vbar[self.t]))

    def _obs(self) -> np.ndarray:
        d = self.day
        private = [1.0 - self.t / d.n, self.remaining / self.Q]
        return np.concatenate([d.features[self.t], private])

    def step(self, frac: float):
        d, t = self.day, self.t
        last = t == d.n - 1
        cap = self.cap()
        if last:
            q = self.remaining
            self.forced += max(0.0, q - cap)
        else:
            q = min(float(np.clip(frac, 0.0, 1.0)) * self.remaining, cap)
        price = float(d.open[t])
        parts = self.costs.fill(q, price, float(d.sigma[t]), float(d.vbar[t]), float(d.range_[t]),
                                bought=self.Q - self.remaining, ref_volume=d.prev_volume)
        parts["timing"] = q * (price - d.arrival)
        scale = 1e4 / (self.Q * d.arrival)
        for k, v in parts.items():
            self.acc[k] += v * scale
        reward = -sum(parts.values()) * scale
        self.fills[t] = q
        self.remaining = 0.0 if last else self.remaining - q
        self.t += 1
        done = last
        obs = None if done else self._obs()
        return obs, reward, done, {"qty": q, "price": price}

    def result(self) -> EpisodeResult:
        return EpisodeResult(self.day.date, sum(self.acc.values()), dict(self.acc),
                             self.forced / self.Q, self.fills.copy())


def run_episode(env: ExecutionEnv, day: Day, policy) -> EpisodeResult:
    obs = env.reset(day)
    policy.start(env)
    done = False
    while not done:
        obs, _, done, _ = env.step(policy.act(obs, env))
    return env.result()
