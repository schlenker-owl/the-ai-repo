#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml  # pip install pyyaml

from airoad.cv.video_analysis import AnalyzerConfig, LineSpec, VideoAnalyzer, ZoneSpec

# ---------------------------
# YAML loading
# ---------------------------


def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------
# Config build helpers
# ---------------------------


def _build_base_cfg(d: Dict[str, Any]) -> AnalyzerConfig:
    """
    Build a base AnalyzerConfig from YAML dict. 'source' may be missing if using source_dir mode.
    We'll override per-file outputs later.
    """
    lines = [LineSpec(**ln) for ln in d.get("lines", [])]
    zones = [ZoneSpec(**z) for z in d.get("zones", [])]

    # allow source to be omitted when using source_dir
    source_val = d.get("source", "")  # will be replaced per-file if blank

    return AnalyzerConfig(
        source=source_val,
        model=d.get("model", "yolo11n.pt"),
        tracker=d.get("tracker", "botsort.yaml"),
        out_video=d.get("out_video", "outputs/cv/analysis.mp4"),
        events_csv=d.get("events_csv", "outputs/cv/events.csv"),
        tracks_csv=d.get("tracks_csv", "outputs/cv/tracks.csv"),
        summary_json=d.get("summary_json", "outputs/cv/summary.json"),
        log_file=d.get("log_file", "outputs/cv/analysis.log"),
        conf=float(d.get("conf", 0.3)),
        iou=float(d.get("iou", 0.5)),
        imgsz=int(d.get("imgsz", 640)),
        classes=d.get("classes"),
        device=d.get("device"),
        frame_stride=int(d.get("frame_stride", 1)),
        draw_labels=bool(d.get("draw_labels", True)),
        draw_masks=bool(d.get("draw_masks", True)),
        draw_tracks=bool(d.get("draw_tracks", True)),
        show=bool(d.get("show", False)),
        save_video=bool(d.get("save_video", True)),
        lines=lines,
        zones=zones,
        meter_per_pixel=d.get("meter_per_pixel"),
        max_speed_kmh=float(d.get("max_speed_kmh", 200.0)),
        speed_ema_alpha=float(d.get("speed_ema_alpha", 0.3)),
        track_lost_patience=int(d.get("track_lost_patience", 30)),
        log_level=str(d.get("log_level", "INFO")),
        log_every_n_frames=int(d.get("log_every_n_frames", 100)),
    )


def _cfg_for_file(base: AnalyzerConfig, src_path: Path, out_root: Path) -> AnalyzerConfig:
    """
    Clone the base config for a single video file and wire per-file output paths.
    Outputs live under: {out_root}/{video-stem}/...
    """
    stem = src_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    return AnalyzerConfig(
        source=str(src_path),
        model=base.model,
        tracker=base.tracker,
        out_video=str(out_dir / "analysis.mp4"),
        events_csv=str(out_dir / "events.csv"),
        tracks_csv=str(out_dir / "tracks.csv"),
        summary_json=str(out_dir / "summary.json"),
        log_file=str(out_dir / "analysis.log"),
        conf=base.conf,
        iou=base.iou,
        imgsz=base.imgsz,
        classes=base.classes,
        device=base.device,
        frame_stride=base.frame_stride,
        draw_labels=base.draw_labels,
        draw_masks=base.draw_masks,
        draw_tracks=base.draw_tracks,
        show=base.show,
        save_video=base.save_video,
        lines=base.lines,
        zones=base.zones,
        meter_per_pixel=base.meter_per_pixel,
        max_speed_kmh=base.max_speed_kmh,
        speed_ema_alpha=base.speed_ema_alpha,
        track_lost_patience=base.track_lost_patience,
        log_level=base.log_level,
        log_every_n_frames=base.log_every_n_frames,
    )


# ---------------------------
# Video discovery
# ---------------------------

_DEFAULT_PATTERNS = ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v", "*.webm"]


def _iter_videos(source_dir: Path, patterns: Iterable[str], recursive: bool) -> List[Path]:
    vids: List[Path] = []
    for pat in patterns:
        if recursive:
            vids.extend(source_dir.rglob(pat))
        else:
            vids.extend(source_dir.glob(pat))
    # dedupe and sort by name for stable order
    uniq = sorted(set([p for p in vids if p.is_file()]))
    return uniq


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    cfg_dict = _load_yaml_cfg(args.config)
    base_cfg = _build_base_cfg(cfg_dict)

    # Batch mode knobs (all optional)
    source_dir = cfg_dict.get("source_dir")  # e.g., "data/videos"
    output_root = Path(cfg_dict.get("output_root", "outputs/cv")).resolve()
    patterns = cfg_dict.get("source_glob", _DEFAULT_PATTERNS)
    if isinstance(patterns, str):
        patterns = [patterns]
    recursive = bool(cfg_dict.get("recursive", False))

    if source_dir:
        # -------- Directory (batch) mode --------
        src_dir = Path(source_dir).expanduser().resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"source_dir not found: {src_dir}")

        videos = _iter_videos(src_dir, patterns, recursive)
        if not videos:
            print(
                f"[video_analysis] No videos matched in {src_dir} with patterns {list(patterns)} (recursive={recursive})"
            )
            return

        batch_summaries: List[Dict[str, Any]] = []
        print(f"[video_analysis] Found {len(videos)} videos. Output root: {output_root}")

        for i, vp in enumerate(videos, 1):
            per_cfg = _cfg_for_file(base_cfg, vp, output_root)
            print(f"[video_analysis] ({i}/{len(videos)}) Analyzing: {vp.name}")
            analyzer = VideoAnalyzer(per_cfg)
            try:
                summary = analyzer.run()
                batch_summaries.append(
                    {
                        "video": str(vp),
                        "outputs": {
                            "out_video": per_cfg.out_video,
                            "events_csv": per_cfg.events_csv,
                            "tracks_csv": per_cfg.tracks_csv,
                            "summary_json": per_cfg.summary_json,
                            "log_file": per_cfg.log_file,
                        },
                        "summary": summary,
                    }
                )
            except Exception as e:
                # Keep going if one file fails
                print(f"[video_analysis] ERROR on {vp}: {e}")

        # Write batch summary
        batch_summary_path = output_root / "_batch_summary.json"
        with open(batch_summary_path, "w") as f:
            json.dump(batch_summaries, f, indent=2)
        print(f"[video_analysis] Batch complete. Wrote {batch_summary_path}")

    else:
        # -------- Single-file mode --------
        if not base_cfg.source:
            raise ValueError(
                "YAML must include either 'source' (file/stream) or 'source_dir' (folder)."
            )
        # ensure output dir for single mode (Analyzer also ensures its own)
        os.makedirs(os.path.dirname(base_cfg.out_video) or ".", exist_ok=True)
        analyzer = VideoAnalyzer(base_cfg)
        summary = analyzer.run()
        print("[video_analysis] Summary:", summary)


if __name__ == "__main__":
    main()
