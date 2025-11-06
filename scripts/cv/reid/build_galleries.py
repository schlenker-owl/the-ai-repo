#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


def _load_nearest_row(tracks: pd.DataFrame, tid: int) -> pd.Series | None:
    """Pick a representative row for a track (mid-frame or max conf)."""
    g = tracks[tracks["track_id"] == tid]
    if g.empty:
        return None
    if "frame" in g.columns:
        fmin = int(g["frame"].min())
        fmax = int(g["frame"].max())
        fmid = (fmin + fmax) // 2
        g = g.iloc[(g["frame"] - fmid).abs().argsort()]  # nearest to mid
        return g.iloc[0]
    # fallback: max conf
    if "conf" in g.columns:
        return g.sort_values("conf", ascending=False).iloc[0]
    return g.iloc[0]


def _crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1 : y2 + 1, x1 : x2 + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/cv", help="Folder with per-video outputs")
    ap.add_argument("--per-id", type=int, default=8, help="Max thumbnails per global id")
    ap.add_argument("--thumb", type=int, default=224, help="Thumbnail square size")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reid_dir = root / "reid"
    gid_map_csv = reid_dir / "global_id_map.csv"
    if not gid_map_csv.exists():
        raise FileNotFoundError(f"Missing {gid_map_csv}; run associate_tracks.py first.")

    gmap = pd.read_csv(gid_map_csv)  # video, track_id, class_name, ..., global_id
    out_root = reid_dir / "galleries"
    out_root.mkdir(parents=True, exist_ok=True)

    # group by global_id
    for gid, g in gmap.groupby("global_id"):
        gal_dir = out_root / f"global_{gid:06d}"
        gal_dir.mkdir(parents=True, exist_ok=True)

        thumbs_done = 0
        # iterate each (video, track_id)
        for _, row in g.iterrows():
            if thumbs_done >= args.per_id:
                break

            vname = str(row["video"])
            tid = int(row["track_id"])
            vdir = root / vname
            tracks_csv = vdir / "tracks.csv"
            video_mp4 = vdir / "analysis.mp4"
            if not tracks_csv.exists() or not video_mp4.exists():
                continue

            tdf = pd.read_csv(tracks_csv)
            rep = _load_nearest_row(tdf, tid)
            if rep is None:
                continue

            frame_idx = int(rep["frame"]) if "frame" in rep else 0
            x1, y1, x2, y2 = map(int, [rep["x1"], rep["y1"], rep["x2"], rep["y2"]])

            cap = cv2.VideoCapture(str(video_mp4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue

            crop = _crop(frame, (x1, y1, x2, y2))
            if crop is None:
                continue

            # resize to square thumbnail
            th = args.thumb
            h, w = crop.shape[:2]
            scale = th / max(h, w)
            crop_r = cv2.resize(
                crop,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.full((th, th, 3), 30, dtype=np.uint8)
            yy = (th - crop_r.shape[0]) // 2
            xx = (th - crop_r.shape[1]) // 2
            canvas[yy : yy + crop_r.shape[0], xx : xx + crop_r.shape[1]] = crop_r

            out_name = gal_dir / f"{vname}_tid{tid}_f{frame_idx}.jpg"
            cv2.imwrite(str(out_name), canvas)
            thumbs_done += 1

        if thumbs_done == 0:
            gal_dir.rmdir()
        else:
            print(f"[galleries] global {gid}: {thumbs_done} thumbs -> {gal_dir}")


if __name__ == "__main__":
    main()
