"""Forecast models. Each takes the history up to the issue month (inclusive) and
returns a forecast for the next `h` months. They never see later data."""
import warnings

import numpy as np
import pandas as pd

from . import config


def _seasonal_cycle(y: pd.Series) -> pd.Series:
    """Mean monthly deviation from a centred 12-month mean, from history only."""
    smooth = y.rolling(12, center=True).mean()
    dev = (y - smooth).dropna()
    cyc = dev.groupby(dev.index.month).mean()
    return cyc - cyc.mean()


def seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    """Baseline: the same month last year (what an operator assumes without a model)."""
    last12 = y.iloc[-12:].to_numpy()
    return np.array([last12[i % 12] for i in range(h)])


def trend_seasonal(y: pd.Series, h: int, window: int = config.TREND_WINDOW) -> np.ndarray:
    """Local linear trend of the deseasonalised level + climatological seasonal cycle."""
    cyc = _seasonal_cycle(y)
    ds = y - y.index.month.map(cyc).to_numpy()
    tail = ds.iloc[-window:]
    x = np.arange(len(tail))
    slope, intercept = np.polyfit(x, tail.to_numpy(), 1)
    level_now = intercept + slope * x[-1]
    months = [(y.index[-1] + k).month for k in range(1, h + 1)]
    return np.array([level_now + slope * k + cyc.loc[m] for k, m in zip(range(1, h + 1), months)])


def sarima(y: pd.Series, h: int) -> np.ndarray:
    """SARIMA(1,1,0)(0,1,1)12 on the last 15 years."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    tail = y.iloc[-180:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(tail.to_numpy(), order=(1, 1, 0), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    return np.asarray(res.forecast(h))


MODELS = {
    "seasonal_naive": seasonal_naive,
    "trend_seasonal": trend_seasonal,
    "sarima": sarima,
}


def ensemble(fc: dict) -> np.ndarray:
    return (fc["trend_seasonal"] + fc["sarima"]) / 2
