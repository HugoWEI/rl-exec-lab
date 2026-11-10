import numpy as np
import pytest

from rlexec import (BuyAtEnd, BuyAtStart, CostModel, ExecutionEnv, Market, PerfectForesight, RandomPolicy,
                    TWAP, chronological_split, evaluate, run_episode, summarize, synthetic_bars)
from rlexec.costs import PermImpact

BARS = synthetic_bars(n_days=12, seed=1)
MARKET = Market(BARS, step_minutes=5, window=60)
POLICIES = [BuyAtStart(), BuyAtEnd(), TWAP(), RandomPolicy(0), PerfectForesight()]


def test_days_built():
    assert len(MARKET.days) == 11  # first day dropped (no previous-day volume / warm-up)
    assert all(d.n == 288 for d in MARKET.days)


def test_rewards_sum_to_minus_shortfall_and_breakdown_adds_up():
    env = ExecutionEnv(MARKET, CostModel())
    day = MARKET.days[3]
    obs, total, done, pol = env.reset(day), 0.0, False, RandomPolicy(3)
    while not done:
        obs, r, done, _ = env.step(pol.act(obs, env))
        total += r
    res = env.result()
    assert total == pytest.approx(-res.is_bps)
    assert sum(res.breakdown_bps.values()) == pytest.approx(res.is_bps)
    assert res.fills.sum() == pytest.approx(env.Q)
    paid = sum(q * p for q, p in zip(res.fills, day.open))  # timing alone reproduces the raw average price
    assert res.breakdown_bps["timing"] == pytest.approx((paid / (env.Q * day.arrival) - 1) * 1e4)


def test_frictionless_limits():
    env = ExecutionEnv(MARKET, CostModel.frictionless())
    day = MARKET.days[2]
    pf = run_episode(env, day, PerfectForesight())
    assert pf.is_bps == pytest.approx((day.open.min() / day.arrival - 1) * 1e4)
    assert run_episode(env, day, TWAP()).is_bps == pytest.approx((day.open.mean() / day.arrival - 1) * 1e4)
    assert run_episode(env, day, BuyAtStart()).is_bps == pytest.approx(0.0)
    assert run_episode(env, day, BuyAtEnd()).is_bps == pytest.approx((day.open[-1] / day.arrival - 1) * 1e4)
    assert all(v == 0 for k, v in pf.breakdown_bps.items() if k != "timing")


@pytest.mark.parametrize("cfg", [
    {},
    {"temp_impact": {"enabled": False}},
    {"participation_cap": {"enabled": False}},
    {"temp_impact": {"delta": 1.0, "eta": 0.5}, "spread": {"mode": "range"}},
])
def test_perfect_foresight_is_a_lower_bound(cfg):
    env = ExecutionEnv(MARKET, CostModel.from_dict(cfg))
    res = evaluate(env, MARKET.days, POLICIES).pivot(index="date", columns="policy", values="is_bps")
    for name in ["buy_at_start", "buy_at_end", "twap", "random"]:
        assert (res["perfect_foresight"] <= res[name] + 1e-6).all(), name


def test_cap_respected_except_forced_final_step():
    costs = CostModel()
    env = ExecutionEnv(MARKET, costs)
    for pol in POLICIES:
        for day in MARKET.days[:4]:
            r = run_episode(env, day, pol)
            cap = costs.cap_qty(day.vbar)
            assert (r.fills[:-1] <= cap[:-1] * (1 + 1e-9)).all()
            if r.forced_pct == 0:
                assert r.fills[-1] <= cap[-1] * (1 + 1e-9)


def test_permanent_impact_charges_and_is_nearly_schedule_independent():
    costs = CostModel.frictionless()
    costs.perm_impact = PermImpact(True, 0.05)
    env = ExecutionEnv(MARKET, costs, {"mode": "pct_prev_day_volume", "value": 0.01})
    r = run_episode(env, MARKET.days[0], TWAP())
    # continuous limit: gamma * (Q/Vref) / 2 = 0.05 * 0.01 / 2 = 2.5 bps
    assert r.breakdown_bps["perm_impact"] == pytest.approx(2.5, rel=0.05)


def test_features_are_causal():
    bars2 = BARS.copy()
    cut = BARS.index[1440 * 6 + 700]  # perturb everything from mid-day 7 onwards
    bars2.loc[bars2.index >= cut, ["open", "high", "low", "close"]] *= 1.05
    bars2.loc[bars2.index >= cut, "volume"] *= 3.0
    m2 = Market(bars2, step_minutes=5, window=60)
    k = 700 // 5  # steps strictly before the perturbed bar
    d1, d2 = MARKET.days[5], m2.days[5]
    assert d1.date == d2.date
    for a, b in [(d1.features, d2.features), (d1.sigma, d2.sigma), (d1.vbar, d2.vbar), (d1.range_, d2.range_)]:
        np.testing.assert_allclose(a[:k], b[:k])
    for e1, e2 in zip(MARKET.days[:5], m2.days[:5]):
        np.testing.assert_allclose(e1.features, e2.features)


def test_split_and_summary():
    s = chronological_split(MARKET.days, 0.5, 0.25, embargo_days=1)
    dates = [d.date for part in ("train", "val", "test") for d in s[part]]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)
    assert s["train"][-1].date < s["val"][0].date - np.timedelta64(1, "D")
    env = ExecutionEnv(MARKET, CostModel())
    summ = summarize(evaluate(env, MARKET.days, POLICIES), n_boot=200)
    assert summ.loc["twap", "vs_twap"] == 0
    assert summ.index[0] == "perfect_foresight"


def test_describe_flags_frictionless():
    assert "WARNING" in CostModel.frictionless().describe()
    assert "WARNING" not in CostModel().describe()
