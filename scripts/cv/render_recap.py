#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


def _load_video_meta(vpath: Path) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vpath}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, n


def _infer_total_frames(meta_frames: int, tracks: pd.DataFrame, events: pd.DataFrame) -> int:
    if meta_frames and meta_frames > 0:
        return meta_frames
    candidates: List[int] = []
    if "frame" in tracks.columns and not tracks.empty:
        candidates.append(int(pd.to_numeric(tracks["frame"], errors="coerce").max() or 0) + 1)
    if "frame" in events.columns and not events.empty:
        candidates.append(int(pd.to_numeric(events["frame"], errors="coerce").max() or 0) + 1)
    return max(candidates) if candidates else 1


def _prep_counts(tracks: pd.DataFrame, events: pd.DataFrame, meta_total_frames: int) -> Dict:
    # Ensure numeric frame column
    if "frame" in tracks.columns:
        tracks["frame"] = pd.to_numeric(tracks["frame"], errors="coerce").fillna(0).astype(int)
    else:
        tracks["frame"] = 0

    total_frames = _infer_total_frames(meta_total_frames, tracks, events)
    total_frames = int(max(1, total_frames))

    # Active tracks per frame
    if {"frame", "track_id"}.issubset(tracks.columns):
        active_grp = tracks.groupby("frame")["track_id"].nunique()
        active_arr = np.zeros(total_frames, dtype=np.int32)
        idx = np.clip(active_grp.index.values, 0, total_frames - 1)
        active_arr[idx] = active_grp.values
    else:
        active_arr = np.zeros(total_frames, dtype=np.int32)

    # Cumulative unique tracks over time
    if {"frame", "track_id"}.issubset(tracks.columns) and not tracks.empty:
        first_frames = tracks.groupby("track_id")["frame"].min().values
        uniq_hits = np.zeros(total_frames, dtype=np.int32)
        ff = np.clip(first_frames, 0, total_frames - 1)
        np.add.at(uniq_hits, ff, 1)
        uniq_cum = np.cumsum(uniq_hits)
    else:
        uniq_cum = np.zeros(total_frames, dtype=np.int32)

    # Zones cumulative
    zones = []
    zone_cum: Dict[str, np.ndarray] = {}
    if {"event", "zone", "frame"}.issubset(events.columns) and not events.empty:
        zone_enters = events[events["event"] == "zone_enter"].copy()
        if not zone_enters.empty:
            zone_enters["frame"] = (
                pd.to_numeric(zone_enters["frame"], errors="coerce").fillna(0).astype(int)
            )
            zones = sorted(zone_enters["zone"].dropna().unique().tolist())
            for z in zones:
                arr = np.zeros(total_frames, dtype=np.int32)
                zf = zone_enters[zone_enters["zone"] == z]["frame"].values
                zf = np.clip(zf, 0, total_frames - 1)
                np.add.at(arr, zf, 1)
                zone_cum[z] = np.cumsum(arr)

    # Line crossings cumulative per line & dir
    lines = []
    line_cum: Dict[str, Dict[str, np.ndarray]] = {}
    if {"event", "line", "line_dir", "frame"}.issubset(events.columns) and not events.empty:
        line_rows = events[events["event"] == "line_cross"].copy()
        if not line_rows.empty:
            line_rows["frame"] = (
                pd.to_numeric(line_rows["frame"], errors="coerce").fillna(0).astype(int)
            )
            lines = sorted(line_rows["line"].dropna().unique().tolist())
            line_dirs = {"A->B", "B->A"}
            for ln in lines:
                line_cum[ln] = {}
                for d in line_dirs:
                    arr = np.zeros(total_frames, dtype=np.int32)
                    f = line_rows[(line_rows["line"] == ln) & (line_rows["line_dir"] == d)][
                        "frame"
                    ].values
                    f = np.clip(f, 0, total_frames - 1)
                    np.add.at(arr, f, 1)
                    line_cum[ln][d] = np.cumsum(arr)

    return {
        "total_frames": total_frames,
        "active": active_arr,
        "uniq_cum": uniq_cum,
        "zone_cum": zone_cum,
        "line_cum": line_cum,
        "zones": zones,
        "lines": lines,
    }


def _draw_panel(frame, fidx: int, fps: float, counts: Dict):
    H, W = frame.shape[:2]
    # semi-transparent panel
    panel_w = int(0.33 * W)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, H), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    x0, y = 12, 28
    tt = f"{(fidx / max(fps, 1e-9)):6.1f}s"
    cv2.putText(
        frame, f"Recap — t={tt}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )
    y += 24

    # Active & Unique
    active = (
        int(counts["active"][min(fidx, len(counts["active"]) - 1)]) if len(counts["active"]) else 0
    )
    uniq = (
        int(counts["uniq_cum"][min(fidx, len(counts["uniq_cum"]) - 1)])
        if len(counts["uniq_cum"])
        else 0
    )
    cv2.putText(
        frame, f"Active tracks: {active}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 240, 255), 2
    )
    y += 22
    cv2.putText(
        frame, f"Unique so far: {uniq}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 240, 180), 2
    )
    y += 28

    # Zones
    if counts["zones"]:
        cv2.putText(frame, "Zones:", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 240, 120), 2)
        y += 22
        for z in counts["zones"]:
            zarr = counts["zone_cum"].get(z, np.zeros(1, dtype=np.int32))
            val = int(zarr[min(fidx, len(zarr) - 1)])
            cv2.putText(
                frame,
                f"  {z}: {val}",
                (x0 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            y += 20
        y += 6

    # Lines
    if counts["lines"]:
        cv2.putText(frame, "Lines:", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 2)
        y += 22
        for ln in counts["lines"]:
            a2b_arr = counts["line_cum"].get(ln, {}).get("A->B", np.zeros(1, dtype=np.int32))
            b2a_arr = counts["line_cum"].get(ln, {}).get("B->A", np.zeros(1, dtype=np.int32))
            a2b = int(a2b_arr[min(fidx, len(a2b_arr) - 1)])
            b2a = int(b2a_arr[min(fidx, len(b2a_arr) - 1)])
            cv2.putText(
                frame,
                f"  {ln}: A→B {a2b} | B→A {b2a}",
                (x0 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            y += 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=str, required=True, help="outputs/cv/<video-stem>")
    ap.add_argument(
        "--out-name", type=str, default="recap.mp4", help="output filename inside video-dir"
    )
    args = ap.parse_args()

    d = Path(args.video_dir).resolve()
    annotated = d / "analysis.mp4"
    tracks_csv = d / "tracks.csv"
    events_csv = d / "events.csv"
    summary_json = d / "summary.json"
    out_video = d / args.out_name

    if not annotated.exists():
        raise FileNotFoundError(f"Missing annotated video: {annotated}")
    if not tracks_csv.exists() or not summary_json.exists():
        raise FileNotFoundError("tracks.csv or summary.json missing.")

    # We no longer read summary_json here to avoid an unused variable warning.

    tracks = pd.read_csv(tracks_csv)
    events = pd.read_csv(events_csv) if events_csv.exists() else pd.DataFrame()

    W, H, FPS, meta_N = _load_video_meta(annotated)
    counts = _prep_counts(tracks, events, meta_total_frames=meta_N)

    cap = cv2.VideoCapture(str(annotated))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, FPS, (W, H))

    fidx = 0
    total_frames = counts["total_frames"]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _draw_panel(frame, fidx, FPS, counts)
        writer.write(frame)
        fidx += 1
        # Guard: if metadata under-reported frames, stop at inferred total_frames
        if fidx >= total_frames:
            break

    cap.release()
    writer.release()
    print(f"[render_recap] Wrote {out_video}")


if __name__ == "__main__":
    main()
