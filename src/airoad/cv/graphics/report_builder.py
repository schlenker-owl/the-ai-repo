from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from .heatmaps import save_activity_heatmap
from .plots import (
    plot_active_tracks_timeseries,
    plot_class_histogram,
    plot_line_dir_bars,
    plot_speed_distribution,
    plot_zone_dwell_hist,
)


def _safe_read_csv(path: Path, columns_if_empty: list[str] | None = None) -> pd.DataFrame:
    """
    Read a CSV safely. If file is missing or empty, return an empty DataFrame
    with the provided columns so downstream code can proceed.
    """
    path = Path(path)
    if (not path.exists()) or (os.path.getsize(path) == 0):
        return pd.DataFrame(columns=columns_if_empty or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns_if_empty or [])


def build_graphics_for_video_dir(video_out_dir: str | Path) -> Dict[str, str]:
    """
    video_out_dir: path like outputs/cv/<video-stem> containing
      - analysis.mp4
      - events.csv (may be empty or missing)
      - tracks.csv
      - summary.json

    Returns a dict of artifact name -> path.
    """
    d = Path(video_out_dir)
    tracks_csv = d / "tracks.csv"
    events_csv = d / "events.csv"
    summary_json = d / "summary.json"
    annotated_mp4 = d / "analysis.mp4"
    figs_dir = d / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Robust reads
    tracks_cols = [
        "frame",
        "track_id",
        "class_id",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "conf",
        "speed_kmh",
        "speed_kmh_ema",
    ]
    events_cols = [
        "frame",
        "time_s",
        "event",
        "zone",
        "line",
        "line_dir",
        "track_id",
        "class_id",
        "class_name",
        "zone_dwell_s",
    ]

    tracks = _safe_read_csv(tracks_csv, tracks_cols)
    events = _safe_read_csv(events_csv, events_cols)

    # Summary (fps, line counts, etc.)
    with open(summary_json, "r") as f:
        summary = json.load(f)
    fps = float(summary.get("fps", 30.0))

    artifacts: Dict[str, str] = {}

    # Charts (each function handles empties gracefully)
    artifacts["classes_bar"] = str(plot_class_histogram(tracks, figs_dir))
    artifacts["active_timeseries"] = str(plot_active_tracks_timeseries(tracks, fps, figs_dir))
    artifacts["zone_dwell_hist"] = str(plot_zone_dwell_hist(events, figs_dir))
    artifacts["line_dir_bars"] = str(plot_line_dir_bars(summary, figs_dir))
    artifacts["speed_distribution"] = str(plot_speed_distribution(tracks, figs_dir))

    # Heatmap overlay (falls back to blank if no coords)
    artifacts["activity_heatmap"] = save_activity_heatmap(
        tracks=tracks,
        out_png=figs_dir / "activity_heatmap.png",
        bg_image_path=annotated_mp4 if annotated_mp4.exists() else None,
        frame_size_hint=None,
        bins=64,
        alpha=0.45,
    )

    # Index of produced figures
    with open(d / "figs" / "_figs_index.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    return artifacts
