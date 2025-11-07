#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import pandas as pd

from airoad.core.io import ensure_dir
from airoad.core.manifest import write_manifest
from airoad.timeseries.anomaly import stl_or_zscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns ds,y")
    ap.add_argument("--out", type=str, default="outputs/ts/anomaly")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds").sort_index().asfreq("D")
    y = df["y"].astype(float)

    res = stl_or_zscore(y)
    out_dir = ensure_dir(args.out)
    out_json = out_dir / "anomalies.json"
    anomalies = [
        {"ds": str(idx), "y": float(row["y"]), "is_anom": bool(row["is_anom"])}
        for idx, row in res.iterrows()
    ]
    out_json.write_text(json.dumps(anomalies, indent=2))
    write_manifest(
        out_dir,
        "ts/anomaly_csv",
        "0.1.0",
        None,
        [args.csv],
        {"anomalies_json": str(out_json)},
        None,
        "cpu",
    )
    print(f"[ts/anomaly_csv] wrote {out_json}")


if __name__ == "__main__":
    main()
