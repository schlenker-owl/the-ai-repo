from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: str | Path) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_class_histogram(tracks: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    fn = Path(out_dir) / "classes_bar.png"
    if "class_name" in tracks:
        s = tracks.groupby("class_name")["track_id"].nunique().sort_values(ascending=False)
    else:
        s = tracks.groupby("class_id")["track_id"].nunique().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    s.plot(kind="bar")
    plt.title("Unique Tracks per Class")
    plt.ylabel("unique tracks")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    return fn


def plot_active_tracks_timeseries(tracks: pd.DataFrame, fps: float, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    fn = Path(out_dir) / "active_tracks_timeseries.png"
    active = tracks.groupby("frame")["track_id"].nunique().sort_index()
    t = active.index.values / max(fps, 1e-9)
    plt.figure(figsize=(8, 3))
    plt.plot(t, active.values, lw=1.5)
    plt.title("Active Tracks Over Time")
    plt.xlabel("time (s)")
    plt.ylabel("active tracks")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    return fn


def plot_zone_dwell_hist(events: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    fn = Path(out_dir) / "zone_dwell_hist.png"
    df = events[events["event"].isin(["zone_exit", "zone_exit_lost"])].copy()
    if "zone_dwell_s" not in df or df.empty:
        # nothing to plot
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No dwell data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(fn, dpi=150)
        plt.close()
        return fn
    plt.figure(figsize=(8, 4))
    for zone, zz in df.groupby("zone"):
        vals = zz["zone_dwell_s"].dropna().astype(float).values
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=20, alpha=0.5, label=str(zone))
    plt.legend()
    plt.title("Zone Dwell (s)")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    return fn


def plot_line_dir_bars(summary: Dict, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    fn = Path(out_dir) / "line_direction_bars.png"
    ldd: Dict[str, Dict[str, int]] = summary.get("line_counts_dir", {})
    if not ldd:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No line crossings", ha="center", va="center")
        plt.axis("off")
        plt.savefig(fn, dpi=150)
        plt.close()
        return fn
    lines = list(ldd.keys())
    a2b = [ldd[k].get("A->B", 0) for k in lines]
    b2a = [ldd[k].get("B->A", 0) for k in lines]
    x = np.arange(len(lines))
    w = 0.35
    plt.figure(figsize=(8, 3))
    plt.bar(x - w / 2, a2b, width=w, label="A→B")
    plt.bar(x + w / 2, b2a, width=w, label="B→A")
    plt.xticks(x, lines, rotation=0)
    plt.ylabel("count")
    plt.title("Line Crossing Directions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    return fn


def plot_speed_distribution(tracks: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    fn = Path(out_dir) / "speed_distribution.png"
    col = (
        "speed_kmh_ema"
        if "speed_kmh_ema" in tracks
        else ("speed_kmh" if "speed_kmh" in tracks else None)
    )
    if col is None or tracks[col].dropna().empty:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No speed data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(fn, dpi=150)
        plt.close()
        return fn
    vals = tracks[col].dropna().astype(float).values
    plt.figure(figsize=(8, 3))
    plt.hist(vals, bins=30)
    plt.title("Speed Distribution (km/h)")
    plt.xlabel("km/h")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    return fn
