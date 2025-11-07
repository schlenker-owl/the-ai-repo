from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    """
    Adds simple calendar features to a DataFrame with a datetime column ds_col.
    """
    df = df.copy()
    t = pd.to_datetime(df[ds_col])
    df["dow"] = t.dt.dayofweek
    df["dom"] = t.dt.day
    df["month"] = t.dt.month
    df["year"] = t.dt.year
    return df
