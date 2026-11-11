"""Bars -> per-day arrays with strictly causal features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BINANCE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def load_binance_klines(paths) -> pd.DataFrame:
    """Load Binance public kline dumps (csv or zip, with or without header) into a UTC-indexed OHLCV frame."""
    if isinstance(paths, (str, bytes)) or not hasattr(paths, "__iter__"):
        paths = [paths]
    frames = []
    for p in sorted(str(x) for x in paths):
        df = pd.read_csv(p, header=None, names=BINANCE_COLS, low_memory=False)
        df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]  # drop header row if present
        ts = df["open_time"].astype("int64").to_numpy()
        ts = np.where(ts > 10**14, ts // 1000, ts)  # some dumps are in microseconds, others milliseconds
        out = df[["open", "high", "low", "close", "volume", "taker_base"]].astype(float)
        out = out.rename(columns={"taker_base": "taker_buy"})  # volume bought by aggressive (taker) buyers
        out.index = pd.to_datetime(ts, unit="ms", utc=True)
        frames.append(out)
    bars = pd.concat(frames).sort_index()
    return bars[~bars.index.duplicated()]


BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{ym}.zip"


def download_binance(symbol: str, start: str, end: str, out="data", log=print) -> list:
    """Download monthly 1-minute kline files (skipping ones already present). Returns the local paths."""
    import urllib.request
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in pd.period_range(start, end, freq="M"):
        url = BINANCE_URL.format(s=symbol, ym=str(month))
        dest = out / url.rsplit("/", 1)[1]
        if not dest.exists():
            try:
                urllib.request.urlretrieve(url, dest)
                log(f"downloaded {dest.name}")
            except Exception as e:  # missing month or network error: report and continue
                log(f"FAILED {dest.name}: {e}")
                dest.unlink(missing_ok=True)
                continue
        paths.append(dest)
    return paths


def synthetic_bars(n_days: int = 120, seed: int = 0, s0: float = 30_000.0, daily_vol: float = 0.03,
                   start: str = "2024-01-01", midday_dip_bps: float = 0.0) -> pd.DataFrame:
    """1-minute bars with a U-shaped intraday vol/volume profile. For tests and demos only.
    With midday_dip_bps = 0 prices are driftless: nothing to predict, no policy should beat TWAP on
    timing. With midday_dip_bps > 0 the price dips by that much around midday on average, a
    deliberately learnable pattern used to check that an agent can learn at all."""
    rng = np.random.default_rng(seed)
    n = 1440 * n_days
    tod = np.arange(n) % 1440
    shape = 1.0 + 0.8 * np.cos(2 * np.pi * tod / 1440) ** 2
    shape /= np.sqrt(np.mean(shape**2))
    sig = daily_vol / np.sqrt(1440) * shape
    drift = -(midday_dip_bps / 1e4) * np.pi / 1440 * np.sin(2 * np.pi * tod / 1440)
    close = s0 * np.exp(np.cumsum(rng.standard_normal(n) * sig + drift))
    open_ = np.concatenate([[s0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * sig / 2)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * sig / 2)
    volume = 10.0 * shape**2 * rng.lognormal(0.0, 0.5, n)
    taker_buy = volume * np.clip(0.5 + 0.1 * rng.standard_normal(n), 0.05, 0.95)  # uninformative order flow
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume,
                         "taker_buy": taker_buy}, index=idx)


# ---- feature registry ------------------------------------------------------------------------
# Every feature at step t uses only bars that closed before t, plus the open of bar t (the price
# the agent trades at). The UI will list these names and descriptions.
def _ret(k):
    return lambda g: np.log(g["open"] / g["open"].shift(k))


def _day(g):
    return g.index.normalize()


def _flow(k):
    """Taker buy/sell imbalance over the last k steps, in [-1, 1] (needs the taker_buy column)."""
    def f(g):
        buy, vol = g["taker_buy"].shift(1).rolling(k).sum(), g["volume"].shift(1).rolling(k).sum()
        return ((2 * buy - vol) / vol).where(vol > 0, 0.0)
    return f


def _range_pos(g):
    """Where the current price sits between the day's low (0) and high (1) so far."""
    hi = g.groupby(_day(g))["high"].transform(lambda x: x.shift(1).cummax())
    lo = g.groupby(_day(g))["low"].transform(lambda x: x.shift(1).cummin())
    return ((g["open"] - lo) / (hi - lo)).clip(-0.5, 1.5).fillna(0.5)


def _vwap_dev(g):
    """log(current price / the day's volume-weighted average price so far)."""
    pv = (g["close"] * g["volume"]).groupby(_day(g)).transform(lambda x: x.shift(1).cumsum())
    v = g.groupby(_day(g))["volume"].transform(lambda x: x.shift(1).cumsum())
    return np.log(g["open"] / (pv / v)).where(v > 0, 0.0).fillna(0.0)


def _vol_ratio(g):
    short = np.log(g["open"] / g["open"].shift(1)).rolling(12).std()
    return (short / g["sigma"]).where(g["sigma"] > 0, 1.0)


FFT_WINDOW = 64


def _fft(g) -> pd.DataFrame:
    """Fourier features of the last FFT_WINDOW log prices (window ends at the current open, so it
    only uses known prices). Each window has its straight-line trend removed first."""
    if "_fft" in g.attrs:
        return g.attrs["_fft"]
    from numpy.lib.stride_tricks import sliding_window_view
    n, w = len(g), FFT_WINDOW
    out = pd.DataFrame(np.nan, index=g.index, columns=["low", "high", "cycle", "smooth_dev"])
    if n >= w:
        win = sliding_window_view(np.log(g["open"].to_numpy()), w)
        tc = np.arange(w) - (w - 1) / 2
        slope = (win * tc).sum(1, keepdims=True) / (tc**2).sum()
        d = win - win.mean(1, keepdims=True) - slope * tc          # detrended window
        spec = np.fft.rfft(d, axis=1)
        power = np.abs(spec) ** 2
        power[:, 0] = 0.0
        total = power.sum(1)
        ok = total > 0
        safe = np.where(ok, total, 1.0)
        scale = np.where(ok, d.std(1), 1.0)
        at_end = np.exp(2j * np.pi * np.arange(spec.shape[1]) * (w - 1) / w)  # evaluates each wave at the last point
        k = power[:, 1:9].argmax(1) + 1                                        # dominant slow wave (period 8 to 64 steps)
        rows = np.arange(len(win))
        cycle = (2.0 / w) * np.real(spec[rows, k] * at_end[k])
        smooth = (2.0 / w) * np.real(spec[:, 1:4] * at_end[1:4]).sum(1)        # slow part (periods over 16 steps)
        vals = np.column_stack([power[:, 1:4].sum(1) / safe, power[:, 16:].sum(1) / safe,
                                cycle / scale, (d[:, -1] - smooth) / scale])
        vals[~ok] = 0.0
        out.iloc[w - 1:] = vals
    g.attrs["_fft"] = out
    return out


FEATURES: dict[str, tuple[str, callable]] = {
    "ret_1": ("log return over the last step", _ret(1)),
    "ret_3": ("log return over the last 3 steps", _ret(3)),
    "ret_12": ("log return over the last 12 steps", _ret(12)),
    "ret_48": ("log return over the last 48 steps", _ret(48)),
    "vol": ("rolling volatility of step returns (sigma)", lambda g: g["sigma"]),
    "rel_volume": ("last step's volume / rolling mean volume", lambda g: g["volume"].shift(1) / g["vbar"]),
    "range": ("rolling mean (high-low)/close", lambda g: g["range"]),
    "price_vs_arrival": ("log(current price / day's arrival price)",
                         lambda g: np.log(g["open"] / g.groupby(g.index.normalize())["open"].transform("first"))),
    "flow_1": ("taker buy/sell imbalance over the last step", _flow(1)),
    "flow_12": ("taker buy/sell imbalance over the last 12 steps", _flow(12)),
    "vol_ratio": ("short-term volatility (12 steps) / rolling volatility", _vol_ratio),
    "range_pos": ("position of the price within the day's range so far (0 = low, 1 = high)", _range_pos),
    "vwap_dev": ("log(current price / day's VWAP so far)", _vwap_dev),
    "fft_low_power": ("share of recent price variation in slow waves (periods over 16 steps), from a Fourier transform of the last 64 prices",
                      lambda g: _fft(g)["low"]),
    "fft_high_power": ("share of recent price variation in fast waves (periods under 4 steps)", lambda g: _fft(g)["high"]),
    "fft_cycle": ("where the price sits in its dominant slow wave (positive = above the wave's middle)",
                  lambda g: _fft(g)["cycle"]),
    "fft_smooth_dev": ("current price minus its Fourier-smoothed path (positive = above the smooth path)",
                       lambda g: _fft(g)["smooth_dev"]),
    "tod_sin": ("time of day, sine", lambda g: np.sin(2 * np.pi * (g.index.hour * 60 + g.index.minute) / 1440)),
    "tod_cos": ("time of day, cosine", lambda g: np.cos(2 * np.pi * (g.index.hour * 60 + g.index.minute) / 1440)),
}
DEFAULT_FEATURES = ["ret_1", "ret_12", "vol", "rel_volume", "price_vs_arrival"]


@dataclass
class Day:
    date: pd.Timestamp
    open: np.ndarray       # trade price at each step (bar open)
    sigma: np.ndarray      # causal rolling vol of step returns
    vbar: np.ndarray       # causal rolling mean step volume
    range_: np.ndarray     # causal rolling mean (high-low)/close
    features: np.ndarray   # (n_steps, n_features), raw (not normalised)
    prev_volume: float     # previous day's total volume

    @property
    def n(self) -> int:
        return len(self.open)

    @property
    def arrival(self) -> float:
        return float(self.open[0])


class Market:
    """Resamples bars to the decision step and slices them into complete UTC days."""

    def __init__(self, bars: pd.DataFrame, step_minutes: int = 5, window: int = 60,
                 features: list[str] | None = None):
        if 1440 % step_minutes:
            raise ValueError("step_minutes must divide 1440")
        self.step_minutes, self.window = step_minutes, window
        self.feature_names = list(features if features is not None else DEFAULT_FEATURES)
        unknown = set(self.feature_names) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")

        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        if "taker_buy" in bars.columns:
            agg["taker_buy"] = "sum"
        elif any(f.startswith("flow_") for f in self.feature_names):
            raise ValueError("flow features need a taker_buy column in the bars")
        g = bars.resample(f"{step_minutes}min").agg(agg)
        g["sigma"] = np.log(g["open"] / g["open"].shift(1)).rolling(window).std()
        g["vbar"] = g["volume"].shift(1).rolling(window).mean()
        g["range"] = ((g["high"] - g["low"]) / g["close"]).shift(1).rolling(window).mean()
        feats = pd.DataFrame({name: FEATURES[name][1](g) for name in self.feature_names}, index=g.index)

        steps_per_day = 1440 // step_minutes
        self.days: list[Day] = []
        prev_date, prev_vol = None, None
        for date, chunk in g.groupby(g.index.normalize()):
            complete = len(chunk) == steps_per_day and chunk[["open", "volume"]].notna().all().all()
            f = feats.loc[chunk.index]
            usable = (complete and prev_vol is not None and prev_date == date - pd.Timedelta(days=1)
                      and chunk[["sigma", "vbar", "range"]].notna().all().all() and f.notna().all().all()
                      and (chunk["vbar"] > 0).all() and prev_vol > 0)
            if usable:
                self.days.append(Day(date, chunk["open"].to_numpy(), chunk["sigma"].to_numpy(),
                                     chunk["vbar"].to_numpy(), chunk["range"].to_numpy(),
                                     f.to_numpy(dtype=float), float(prev_vol)))
            prev_date, prev_vol = (date, chunk["volume"].sum()) if complete else (None, None)


def chronological_split(days: list, train: float = 0.6, val: float = 0.2, embargo_days: int = 1) -> dict[str, list]:
    """Train / validation / test by day, in time order, dropping `embargo_days` at each boundary
    (rolling features look back across midnight, so adjacent days are not independent)."""
    n = len(days)
    a, b = int(n * train), int(n * (train + val))
    return {"train": days[:a], "val": days[a + embargo_days:b], "test": days[b + embargo_days:]}
