from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from airoad.timeseries.anomaly import stl_or_zscore
from airoad.timeseries.eval_ts import mape, smape
from airoad.timeseries.forecast import ets_or_naive


def _make_series(n: int = 45) -> pd.Series:
    start = datetime(2024, 1, 1)
    idx = pd.date_range(start, periods=n, freq="D")
    # mild seasonality + noise
    vals = 10 + 2 * np.sin(np.linspace(0, 3.14 * 2, n)) + 0.5 * np.random.randn(n)
    return pd.Series(vals, index=idx)


def test_forecast_smoke():
    y = _make_series(60)
    horizon = 7
    res = ets_or_naive(y, horizon=horizon)
    assert len(res.yhat) == horizon
    # backtest a quick score (non-NaN)
    score = mape(y.iloc[-horizon:], res.yhat.iloc[:horizon])
    assert np.isfinite(score)


def test_anomaly_smoke():
    y = _make_series(40)
    # Inject an outlier
    y.iloc[10] += 8.0
    anom = stl_or_zscore(y, z=3.0)
    assert "is_anom" in anom.columns
    # Expect at least one anomaly flagged
    assert bool(anom["is_anom"].sum() >= 1)
    # sanity metric
    s = smape(y.iloc[-7:], y.iloc[-7:].rolling(2, min_periods=1).mean())
    assert np.isfinite(s)
