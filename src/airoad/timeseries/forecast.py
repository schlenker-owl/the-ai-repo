from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ForecastResult:
    yhat: pd.Series
    yhat_lower: pd.Series
    yhat_upper: pd.Series


def naive_last(y: pd.Series, horizon: int) -> ForecastResult:
    last = float(y.iloc[-1])
    idx = pd.date_range(y.index[-1], periods=horizon + 1, freq=y.index.freq or "D")[1:]
    yhat = pd.Series([last] * horizon, index=idx)
    return ForecastResult(yhat=yhat, yhat_lower=yhat * 0.95, yhat_upper=yhat * 1.05)


def ets_or_naive(y: pd.Series, horizon: int) -> ForecastResult:
    """
    Try statsmodels ETS; fall back to naive last.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

        seasonal = "add" if (y.index.freqstr or "D") in ("D", "W") else None
        mdl = ExponentialSmoothing(
            y, trend="add", seasonal=seasonal, seasonal_periods=7 if seasonal else None
        )
        fit = mdl.fit()
        f = fit.forecast(horizon)
        return ForecastResult(yhat=f, yhat_lower=f * 0.9, yhat_upper=f * 1.1)
    except Exception:
        return naive_last(y, horizon)
