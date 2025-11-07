#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from airoad.core.io import ensure_dir
from airoad.core.manifest import write_manifest
from airoad.timeseries.eval_ts import mape, smape
from airoad.timeseries.forecast import ets_or_naive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ts/forecast.yaml")
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns ds,y")
    ap.add_argument("--out", type=str, default="outputs/ts/forecast")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    horizon = int(cfg.get("horizon", 14))

    df = pd.read_csv(args.csv)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds").sort_index()
    df = df.asfreq("D")
    y = df["y"].astype(float)

    res = ets_or_naive(y, horizon=horizon)
    out_dir = ensure_dir(args.out)
    fc_path = out_dir / "forecast.csv"
    res.yhat.rename("yhat").to_frame().to_csv(fc_path)

    # backtest (last horizon)
    if len(y) > horizon:
        y_true = y.iloc[-horizon:]
        y_pred = res.yhat.iloc[:horizon]
        metrics = {"mape": mape(y_true, y_pred), "smape": smape(y_true, y_pred)}
    else:
        metrics = {"mape": None, "smape": None}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_manifest(
        out_dir,
        "ts/forecast_csv",
        "0.1.0",
        args.config,
        [args.csv],
        {"forecast_csv": str(fc_path)},
        None,
        "cpu",
    )
    print(f"[ts/forecast_csv] wrote {fc_path} and metrics.json")


if __name__ == "__main__":
    main()
