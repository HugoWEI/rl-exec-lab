"""Build the market, cost model, environment and splits from a config dict."""
from __future__ import annotations

from pathlib import Path

import yaml

from .costs import CostModel
from .data import Market, chronological_split
from .env import ExecutionEnv

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path=None) -> dict:
    return yaml.safe_load(open(path or DEFAULT_CONFIG))


def build(cfg: dict, bars, reward: str = "shortfall"):
    market = Market(bars, **cfg["market"])
    costs = CostModel.from_dict(cfg["costs"])
    env = ExecutionEnv(market, costs, cfg["parent"], cfg.get("action"), reward)
    splits = chronological_split(market.days, **cfg["split"])
    return market, costs, env, splits
