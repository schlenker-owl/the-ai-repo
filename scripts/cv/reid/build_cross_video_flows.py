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
    gid_map_csv = reid_dir / "global_id_map.csv"
    if not gid_map_csv.exists():
        raise FileNotFoundError(f"Missing {gid_map_csv}")

    gmap = pd.read_csv(gid_map_csv)  # columns include: video, global_id, f_min, f_max, class_name
    if gmap.empty:
        print("[flows] global_id_map.csv is empty.")
        return

    # For ordering within a global_id across videos, use earliest f_min
    # Build flows from earlier video -> later video (by f_min order); ignore self-loops
    flows = {}
    for gid, g in gmap.groupby("global_id"):
        # video-level first appearance
        vid_first = g.groupby("video")["f_min"].min().reset_index()
        vid_first = vid_first.sort_values("f_min")
        vids = vid_first["video"].tolist()
        if len(vids) < 2:
            continue
        # create directed edges along this order
        for i in range(len(vids) - 1):
            a, b = vids[i], vids[i + 1]
            if a == b:
                continue
            flows[(a, b)] = flows.get((a, b), 0) + 1

    if not flows:
        print("[flows] No multi-video transitions found.")
        return

    # write CSV + JSON
    out_dir = reid_dir
    rows = [{"from": a, "to": b, "count": c} for (a, b), c in sorted(flows.items())]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "flows.csv", index=False)
    with open(out_dir / "flows.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[flows] wrote {out_dir/'flows.csv'} and flows.json")


if __name__ == "__main__":
    main()
