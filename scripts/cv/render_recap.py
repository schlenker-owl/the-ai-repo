#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=str, required=True, help="outputs/cv/<video-stem>")
    ap.add_argument(
        "--out-name", type=str, default="recap.mp4", help="output filename inside video-dir"
    )
    args = ap.parse_args()

    d = Path(args.video_dir).resolve()
    annotated = d / "analysis.mp4"
    out_video = d / args.out_name

    if not annotated.exists():
        raise FileNotFoundError(f"Missing annotated video: {annotated}")

    cap = cv2.VideoCapture(str(annotated))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {annotated}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # no overlay; write as-is
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[render_recap] Wrote {out_video}")


if __name__ == "__main__":
    main()
