"""Rolling-origin hindcast: issue a forecast every month as if it were that month.

Prediction intervals and crossing probabilities at issue month t use only forecast
errors that had already been verified by month t (strictly no look-ahead).
"""
import numpy as np
import pandas as pd

from . import config, models


def run(df: pd.DataFrame) -> pd.DataFrame:
    y = df["level_bs"]
    origins = [p for p in y.index if p >= pd.Period(config.FIRST_ORIGIN, "M") and p < y.index[-1]]
    recs = []
    for t in origins:
        hist = y.loc[:t]
        fc = {name: f(hist, config.HORIZON) for name, f in models.MODELS.items()}
        fc["ensemble"] = models.ensemble(fc)
        for name, vals in fc.items():
            for k, v in enumerate(vals, start=1):
                target = t + k
                actual = y.get(target, np.nan)
                recs.append((t, name, k, target, float(v), float(actual)))
    out = pd.DataFrame(recs, columns=["origin", "model", "h", "target", "forecast", "actual"])
    out["error"] = out["actual"] - out["forecast"]
    return out


class ErrorPool:
    """Verified errors per (model, horizon), indexed by the month they became known.

    For issue month t we take the most recent `window` errors whose target month
    is <= t. A rolling window keeps the intervals honest when the regime changes
    (the decline sped up after 2010 and again after 2020)."""

    def __init__(self, fc: pd.DataFrame, window: int = config.ERROR_WINDOW):
        self.window = window
        self.pool = {}
        v = fc.dropna(subset=["error"])
        for (m, h), g in v.groupby(["model", "h"]):
            g = g.sort_values("target")
            self.pool[(m, h)] = (g["target"].map(lambda p: p.ordinal).to_numpy(), g["error"].to_numpy())

    def get(self, model: str, t: pd.Period, h: int) -> np.ndarray:
        if (model, h) not in self.pool:
            return np.array([])
        tg, err = self.pool[(model, h)]
        n = np.searchsorted(tg, t.ordinal, side="right")
        return err[max(0, n - self.window): n]


def add_intervals(fc: pd.DataFrame, model: str = "ensemble", pool: ErrorPool = None,
                  min_errors: int = 24, adaptive: bool = True) -> pd.DataFrame:
    """80% intervals = forecast + scale * quantiles of past errors.

    With `adaptive`, the scale follows adaptive conformal inference (Gibbs & Candès 2021),
    with delayed feedback: when a forecast's target month arrives, a miss widens the
    scale for that horizon, a hit narrows it slightly. Only already-verified forecasts
    move the scale, so month t never uses information from after t."""
    pool = pool or ErrorPool(fc)
    target_miss = 1 - (config.PI_LEVELS[1] - config.PI_LEVELS[0])
    sub = fc[fc.model == model].copy().sort_values(["origin", "h"])
    scale = {h: 1.0 for h in sub.h.unique()}
    pending = {}  # target month -> list of (h, lo, hi, actual)
    lo_l, hi_l, sc_l = [], [], []
    for t, g in sub.groupby("origin", sort=True):
        if adaptive:
            for h, lo, hi, act in pending.pop(t, []):
                miss = float(not (lo <= act <= hi))
                scale[h] = float(np.clip(scale[h] * np.exp(config.ACI_GAMMA * (miss - target_miss)), 0.5, 5.0))
        for row in g.itertuples():
            e = pool.get(model, t, row.h)
            if len(e) < min_errors:
                lo_l.append(np.nan); hi_l.append(np.nan); sc_l.append(np.nan); continue
            s = scale[row.h] if adaptive else 1.0
            lo = row.forecast + s * np.quantile(e, config.PI_LEVELS[0])
            hi = row.forecast + s * np.quantile(e, config.PI_LEVELS[1])
            lo_l.append(lo); hi_l.append(hi); sc_l.append(s)
            if not np.isnan(row.actual):
                pending.setdefault(row.target, []).append((row.h, lo, hi, row.actual))
    sub["pi_lo"], sub["pi_hi"], sub["scale"] = lo_l, hi_l, sc_l
    return sub


def crossing_probability(fc: pd.DataFrame, threshold: float, model: str = "ensemble",
                         pool: ErrorPool = None, min_errors: int = 24,
                         scales: pd.DataFrame = None) -> pd.DataFrame:
    """For each issue month: P(level < threshold in at least one of the next HORIZON months),
    taken as the max over horizons of the empirical probability (past errors only,
    multiplied by the adaptive scale in force at that month when `scales` is given)."""
    pool = pool or ErrorPool(fc)
    sub = fc[fc.model == model]
    sc = None
    if scales is not None:
        sc = scales.set_index(["origin", "h"])["scale"]
    rows = []
    for t, g in sub.groupby("origin"):
        probs = []
        for row in g.itertuples():
            e = pool.get(model, t, row.h)
            if len(e) < min_errors:
                probs = None; break
            s = 1.0 if sc is None else sc.get((t, row.h), 1.0)
            if s != s:  # NaN
                s = 1.0
            probs.append(float(np.mean(row.forecast + s * e < threshold)))
        if probs is None:
            rows.append((t, np.nan, np.nan)); continue
        k = int(np.argmax(probs))
        rows.append((t, max(probs), k + 1))
    return pd.DataFrame(rows, columns=["origin", "p_cross", "h_at_max"]).set_index("origin")


def skill(fc: pd.DataFrame) -> pd.DataFrame:
    s = fc.dropna(subset=["error"]).assign(ae=lambda d: d.error.abs())
    mae = s.pivot_table(index="h", columns="model", values="ae", aggfunc="mean")
    for m in mae.columns:
        if m != "seasonal_naive":
            mae[f"skill_{m}"] = 1 - mae[m] / mae["seasonal_naive"]
    return mae
