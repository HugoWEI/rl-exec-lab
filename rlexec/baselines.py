"""Reference policies. A policy has start(env) and act(obs, env) -> fraction of remaining inventory."""
from __future__ import annotations

import numpy as np

from .costs import CostModel


class Policy:
    name = "policy"
    uses_future = False  # True only for hindsight benchmarks

    def start(self, env):
        pass

    def act(self, obs, env) -> float:
        raise NotImplementedError


class BuyAtStart(Policy):
    """Buy everything immediately. With a participation cap this means buying at the cap until done."""
    name = "buy_at_start"

    def act(self, obs, env):
        return 1.0


class BuyAtEnd(Policy):
    """Wait as long as possible. With a cap, start as late as the cap allows (estimated from the
    current step's cap, so a small forced remainder is possible)."""
    name = "buy_at_end"

    def act(self, obs, env):
        steps_after = env.day.n - env.t - 1
        if env.remaining > env.cap() * steps_after:
            return 1.0
        return 0.0


class TWAP(Policy):
    """Equal quantity at every step."""
    name = "twap"

    def act(self, obs, env):
        return 1.0 / (env.day.n - env.t)


class RandomPolicy(Policy):
    """Uniform random choice from the discrete action set. A reference for what luck looks like."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return env.action_frac(int(self.rng.integers(env.n_actions)))


class PerfectForesight(Policy):
    """Hindsight-optimal schedule under the same cost model, with continuous quantities.

    Exact lower bound on cost when permanent impact is off (the problem is convex and separable).
    With permanent impact on, the schedule is computed ignoring it and then evaluated with it,
    so it is an approximation rather than a strict bound.
    """
    name = "perfect_foresight"
    uses_future = True

    def start(self, env):
        d = env.day
        self.schedule = optimal_schedule(env.costs, d.open, d.sigma, d.vbar, d.range_, env.Q)

    def act(self, obs, env):
        if env.remaining <= 0:
            return 0.0
        return min(1.0, self.schedule[env.t] / env.remaining)


def optimal_schedule(costs: CostModel, price, sigma, vbar, range_, Q: float) -> np.ndarray:
    """Minimise sum_t [a_t q_t + b_t q_t^(1+delta)] s.t. sum q = Q, 0 <= q_t <= cap_t."""
    price, sigma, vbar = (np.asarray(x, float) for x in (price, sigma, vbar))
    a = price * (1.0 + costs.fee_frac() + costs.spread_frac(np.asarray(range_, float)))
    cap = np.minimum(np.asarray(costs.cap_qty(vbar), float), Q)
    q = np.zeros_like(price)

    if cap.sum() <= Q:  # cap too tight to finish: fill every cap, remainder is forced on the last step
        q = cap.copy()
        q[-1] += Q - cap.sum()
        return q

    t = costs.temp_impact
    if not t.enabled or t.eta <= 0:  # linear cost: fill the cheapest bars first
        left = Q
        for i in np.argsort(a, kind="stable"):
            q[i] = min(cap[i], left)
            left -= q[i]
            if left <= 0:
                break
        return q

    b = price * t.eta * sigma / vbar**t.delta
    k = b * (1.0 + t.delta)

    def alloc(lam):
        return np.minimum(np.maximum((lam - a) / k, 0.0) ** (1.0 / t.delta), cap)

    lo, hi = a.min(), (a + k * cap**t.delta).max()
    for _ in range(200):  # bisection on the marginal cost
        mid = 0.5 * (lo + hi)
        if alloc(mid).sum() < Q:
            lo = mid
        else:
            hi = mid
    q = alloc(hi)
    return q * (Q / q.sum())
