from __future__ import annotations

import json
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


def build_graphics_for_video_dir(video_out_dir: str | Path) -> Dict[str, str]:
    """
    video_out_dir: path like outputs/cv/<video-stem> containing
      - analysis.mp4
      - events.csv
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

    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing tracks.csv in {d}")
    if not events_csv.exists():
        # create empty if not present (older runs)
        events = pd.DataFrame(columns=["event"])
    else:
        events = pd.read_csv(events_csv)

    tracks = pd.read_csv(tracks_csv)
    with open(summary_json, "r") as f:
        summary = json.load(f)
    fps = float(summary.get("fps", 30.0))

    artifacts: Dict[str, str] = {}

    # charts
    artifacts["classes_bar"] = str(plot_class_histogram(tracks, figs_dir))
    artifacts["active_timeseries"] = str(plot_active_tracks_timeseries(tracks, fps, figs_dir))
    artifacts["zone_dwell_hist"] = str(plot_zone_dwell_hist(events, figs_dir))
    artifacts["line_dir_bars"] = str(plot_line_dir_bars(summary, figs_dir))
    artifacts["speed_distribution"] = str(plot_speed_distribution(tracks, figs_dir))

    # heatmap overlay
    artifacts["activity_heatmap"] = save_activity_heatmap(
        tracks=tracks,
        out_png=figs_dir / "activity_heatmap.png",
        bg_image_path=annotated_mp4 if annotated_mp4.exists() else None,
        frame_size_hint=None,
        bins=64,
        alpha=0.45,
    )

    # Write a small index file of figure paths
    with open(d / "figs" / "_figs_index.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    return artifacts
