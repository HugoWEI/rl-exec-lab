"""Config loading, command-line overrides, data resolution and environment construction."""
from __future__ import annotations

from pathlib import Path

import yaml

from .costs import CostModel
from .data import FEATURES, Market, chronological_split, download_binance, load_binance_klines, synthetic_bars
from .env import ExecutionEnv

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path=None) -> dict:
    """Load a config; anything missing falls back to configs/default.yaml."""
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    if path:
        for key, value in (yaml.safe_load(open(path)) or {}).items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = _merge(cfg[key], value)
            else:
                cfg[key] = value
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def add_common_args(ap):
    ap.add_argument("--config", default=None, help="yaml config; missing keys use configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic bars instead of Binance data")
    ap.add_argument("--synthetic-days", type=int, default=None)
    ap.add_argument("--midday-dip-bps", type=float, default=None,
                    help="synthetic only: plant a learnable average midday price dip of this size")
    ap.add_argument("--symbol", default=None, help="Binance spot symbol, e.g. BTCUSDT")
    ap.add_argument("--start", default=None, help="first month, YYYY-MM")
    ap.add_argument("--end", default=None, help="last month, YYYY-MM")
    ap.add_argument("--features", default=None,
                    help="comma-separated state features, or 'all'. Available: " + ", ".join(FEATURES))
    ap.add_argument("--reveal-test", action="store_true", help="report the test split instead of validation")
    ap.add_argument("--label", default=None, help="dataset label shown in the viewer")
    return ap


def config_from_args(args) -> dict:
    cfg = load_config(args.config)
    d = cfg["data"]
    if args.synthetic:
        d["source"] = "synthetic"
    for key in ("synthetic_days", "midday_dip_bps", "symbol", "start", "end"):
        if getattr(args, key, None) is not None:
            d[key] = getattr(args, key)
    if args.features:
        cfg["market"]["features"] = list(FEATURES) if args.features == "all" else args.features.split(",")
    return cfg


def resolve_bars(cfg: dict, log=print):
    """Returns (bars, label, note) for the config's data section, downloading Binance files if needed."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        dip = float(d.get("midday_dip_bps") or 0)
        bars = synthetic_bars(n_days=int(d.get("synthetic_days", 120)), midday_dip_bps=dip)
        if dip:
            return bars, f"Synthetic bars with a {dip:g} bps average midday dip (demo)", (
                "Synthetic prices with a built-in average midday dip: a learnable pattern, used to check "
                "that an agent can learn. Not evidence about real markets.")
        return bars, "Synthetic driftless bars (demo)", (
            "Synthetic, driftless prices: nothing is predictable, so no policy should beat TWAP "
            "except perfect foresight.")
    if d["source"] != "binance":
        raise ValueError(f"unknown data source {d['source']!r}")
    files = download_binance(d["symbol"], str(d["start"]), str(d["end"]), d.get("dir", "data"), log)
    if not files:
        raise SystemExit("no data files available: check the symbol, the months and your network connection")
    return (load_binance_klines(files),
            f"{d['symbol']} spot, 1-minute bars from Binance, {d['start']} to {d['end']}", "")


def build(cfg: dict, bars, reward: str = "shortfall"):
    market = Market(bars, **cfg["market"])
    costs = CostModel.from_dict(cfg["costs"])
    env = ExecutionEnv(market, costs, cfg["parent"], cfg.get("action"), reward)
    splits = chronological_split(market.days, **cfg["split"])
    return market, costs, env, splits
