"""Configurable, transparent transaction-cost model.

Every component can be switched on/off independently. All costs are expressed as a
fraction of the fill price and applied per child order, and every fill returns a
breakdown so results can be attributed component by component.

Inputs (sigma, V, range) are *causal*: they are rolling statistics known at decision
time, so the agent can in principle observe everything the cost model uses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

COMPONENTS = ("fees", "spread", "temp_impact", "perm_impact")


@dataclass
class Fees:
    enabled: bool = True
    bps: float = 1.0  # exchange/broker fee, bps of notional


@dataclass
class Spread:
    enabled: bool = True
    mode: str = "fixed"  # "fixed" | "range"
    half_spread_bps: float = 0.5  # used when mode == "fixed"
    range_coef: float = 0.25  # mode == "range": half-spread = coef * rolling mean (high-low)/close


@dataclass
class TempImpact:
    enabled: bool = True
    eta: float = 1.0  # scale (order-of-magnitude assumption, not calibrated)
    delta: float = 0.5  # 0.5 = square-root law, 1.0 = linear


@dataclass
class PermImpact:
    enabled: bool = False
    gamma: float = 0.05  # price shift (fraction) per unit of (quantity already bought / previous-day volume)


@dataclass
class ParticipationCap:
    enabled: bool = True
    max_pct: float = 0.10  # child order <= max_pct * V (rolling mean step volume)


@dataclass
class CostModel:
    fees: Fees = field(default_factory=Fees)
    spread: Spread = field(default_factory=Spread)
    temp_impact: TempImpact = field(default_factory=TempImpact)
    perm_impact: PermImpact = field(default_factory=PermImpact)
    participation_cap: ParticipationCap = field(default_factory=ParticipationCap)

    # ---- construction / serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict | None) -> "CostModel":
        d = d or {}
        return cls(
            fees=Fees(**d.get("fees", {})),
            spread=Spread(**d.get("spread", {})),
            temp_impact=TempImpact(**d.get("temp_impact", {})),
            perm_impact=PermImpact(**d.get("perm_impact", {})),
            participation_cap=ParticipationCap(**d.get("participation_cap", {})),
        )

    @classmethod
    def frictionless(cls) -> "CostModel":
        return cls(Fees(False), Spread(False), TempImpact(False), PermImpact(False), ParticipationCap(False))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def any_cost_enabled(self) -> bool:
        return any(getattr(self, c).enabled for c in COMPONENTS)

    # ---- per-unit cost fractions --------------------------------------------------------
    def fee_frac(self) -> float:
        return self.fees.bps / 1e4 if self.fees.enabled else 0.0

    def spread_frac(self, range_):
        """Half-spread as a fraction of price. `range_` may be a scalar or array."""
        if not self.spread.enabled:
            return 0.0 * range_
        if self.spread.mode == "fixed":
            return 0.0 * range_ + self.spread.half_spread_bps / 1e4
        if self.spread.mode == "range":
            return self.spread.range_coef * range_
        raise ValueError(f"unknown spread mode {self.spread.mode!r}")

    def temp_frac(self, q: float, sigma: float, vbar: float) -> float:
        if not self.temp_impact.enabled or q <= 0:
            return 0.0
        return self.temp_impact.eta * sigma * (q / vbar) ** self.temp_impact.delta

    def perm_frac(self, bought: float, ref_volume: float) -> float:
        if not self.perm_impact.enabled:
            return 0.0
        return self.perm_impact.gamma * bought / ref_volume

    def cap_qty(self, vbar):
        """Max child-order size for a step (inf when the cap is disabled)."""
        if not self.participation_cap.enabled:
            return float("inf") + 0.0 * vbar
        return self.participation_cap.max_pct * vbar

    # ---- fills ----------------------------------------------------------------------------
    def fill(self, q: float, price: float, sigma: float, vbar: float, range_: float,
             bought: float, ref_volume: float) -> dict[str, float]:
        """Cost of buying `q` at `price`, in quote currency, split by component."""
        notional = q * price
        return {
            "fees": notional * self.fee_frac(),
            "spread": notional * float(self.spread_frac(range_)),
            "temp_impact": notional * self.temp_frac(q, sigma, vbar),
            "perm_impact": notional * self.perm_frac(bought, ref_volume),
        }

    # ---- transparency -----------------------------------------------------------------------
    def describe(self, example_sigma_bps: float = 5.0, example_participation: float = 0.05) -> str:
        on = lambda c: "[ON ]" if c.enabled else "[OFF]"
        f, s, t, p, c = self.fees, self.spread, self.temp_impact, self.perm_impact, self.participation_cap
        spread_txt = (f"fixed half-spread = {s.half_spread_bps:g} bps" if s.mode == "fixed"
                      else f"half-spread = {s.range_coef:g} x rolling mean (high-low)/close")
        lines = [
            "Cost model - each cost is a fraction of the fill price, charged per child order.",
            "sigma = rolling std of step log-returns; V = rolling mean step volume (both use past bars only).",
            f"  {on(f)} fees         : {f.bps:g} bps of notional",
            f"  {on(s)} spread       : {spread_txt}",
            f"  {on(t)} temp impact  : eta * sigma * (q/V)^delta, eta={t.eta:g}, delta={t.delta:g}"
            "  (eta is an assumption, not calibrated)",
            f"  {on(p)} perm impact  : later prices shift by gamma * (qty already bought / prev-day volume),"
            f" gamma={p.gamma:g}  (counterfactual: history never saw these trades)",
            f"  {on(c)} particip. cap: q <= {c.max_pct:.0%} of V; waived on the final step"
            " (forced completion, reported as forced_pct)",
        ]
        sig = example_sigma_bps / 1e4
        ex_temp = self.temp_frac(example_participation, sig, 1.0) * 1e4
        ex_spread = float(self.spread_frac(0.0)) * 1e4 if s.mode == "fixed" else float("nan")
        ex_fee = self.fee_frac() * 1e4
        total = ex_fee + (0 if ex_spread != ex_spread else ex_spread) + ex_temp
        lines.append(
            f"  Worked example: sigma={example_sigma_bps:g} bps, q/V={example_participation:.0%} -> "
            f"fees {ex_fee:.2f} + spread {'n/a (range mode)' if ex_spread != ex_spread else f'{ex_spread:.2f}'}"
            f" + temp impact {ex_temp:.2f} = {total:.2f} bps per unit bought"
        )
        if not self.any_cost_enabled:
            lines.append("  WARNING: all costs are off. The task is pure price timing: perfect foresight buys "
                         "everything at the day's low and results say nothing about execution quality.")
        return "\n".join(lines)
