from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(1e-9, np.abs(yt))
    return float(np.mean(np.abs(yt - yp) / denom))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(1e-9, (np.abs(yt) + np.abs(yp)) / 2.0)
    return float(np.mean(np.abs(yt - yp) / denom))
