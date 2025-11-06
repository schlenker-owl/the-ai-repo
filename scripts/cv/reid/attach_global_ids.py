#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/cv", help="Folder with per-video outputs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reid_dir = root / "reid"
    gid_map = reid_dir / "global_id_map.csv"
    if not gid_map.exists():
        raise FileNotFoundError(f"Missing {gid_map}. Run associate_tracks.py first.")

    gmap = pd.read_csv(gid_map)  # columns: video, track_id, class_id, class_name, ..., global_id
    # process each video folder
    video_dirs = [p for p in root.glob("*") if p.is_dir() and (p / "tracks.csv").exists()]
    if not video_dirs:
        print(f"[attach_global_ids] No videos found under {root}")
        return

    for vdir in video_dirs:
        tracks_csv = vdir / "tracks.csv"
        if not tracks_csv.exists():
            continue
        df = pd.read_csv(tracks_csv)
        # Merge on (video, track_id)
        join_key = ["video", "track_id"]
        if "video" not in df.columns:
            # add a 'video' column equal to folder name (keeps it explicit)
            df["video"] = vdir.name
        merged = df.merge(gmap[["video", "track_id", "global_id"]], on=join_key, how="left")

        out_csv = vdir / "tracks_global.csv"
        merged.to_csv(out_csv, index=False)
        print(f"[attach_global_ids] wrote {out_csv}")

        # augment summary with global stats
        summary_json = vdir / "summary.json"
        summary_global = vdir / "summary_global.json"
        base = {}
        if summary_json.exists():
            with open(summary_json, "r") as f:
                base = json.load(f)

        # compute global unique by class
        guniq = (
            merged.dropna(subset=["global_id"])
            .groupby("class_name")["global_id"]
            .nunique()
            .to_dict()
        )
        out_sum = dict(base)
        out_sum["reid_attached"] = True
        out_sum["global_unique_by_class"] = guniq

        with open(summary_global, "w") as f:
            json.dump(out_sum, f, indent=2)
        print(f"[attach_global_ids] wrote {summary_global}")


if __name__ == "__main__":
    main()
