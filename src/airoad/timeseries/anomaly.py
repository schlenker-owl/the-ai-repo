from __future__ import annotations

import pandas as pd


def zscore_anomalies(y: pd.Series, z: float = 3.0) -> pd.DataFrame:
    mu = float(y.mean())
    sd = float(y.std(ddof=0) + 1e-9)
    zs = (y - mu) / sd
    return pd.DataFrame({"y": y, "z": zs, "is_anom": (zs.abs() >= z)})


def stl_or_zscore(y: pd.Series, z: float = 3.0) -> pd.DataFrame:
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore

        res = STL(y, robust=True, period=7).fit()
        resid = res.resid
        mu = float(resid.mean())
        sd = float(resid.std(ddof=0) + 1e-9)
        anom = (resid - mu).abs() >= z * sd
        return pd.DataFrame({"y": y, "resid": resid, "is_anom": anom})
    except Exception:
        return zscore_anomalies(y, z=z)
