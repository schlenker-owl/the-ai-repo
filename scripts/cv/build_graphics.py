#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from airoad.cv.graphics.report_builder import build_graphics_for_video_dir


def _find_video_dirs(root: Path) -> List[Path]:
    # discover per-video folders by the presence of summary.json
    out: List[Path] = []
    for p in root.glob("*"):
        if p.is_dir() and (p / "summary.json").exists():
            out.append(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=str, help="outputs/cv/<video-stem> to build figures for")
    ap.add_argument(
        "--batch-root",
        type=str,
        default="outputs/cv",
        help="parent folder; build for all subfolders with summary.json",
    )
    args = ap.parse_args()

    artifacts_list = []
    if args.video_dir:
        artifacts = build_graphics_for_video_dir(args.video_dir)
        artifacts_list.append({"video_dir": args.video_dir, "artifacts": artifacts})
    else:
        root = Path(args.batch_root).resolve()
        dirs = _find_video_dirs(root)
        if not dirs:
            print(f"[build_graphics] No per-video folders found under {root}")
            return
        for d in dirs:
            print(f"[build_graphics] Building figures for: {d.name}")
            artifacts = build_graphics_for_video_dir(d)
            artifacts_list.append({"video_dir": str(d), "artifacts": artifacts})

    # batch index
    if artifacts_list:
        idx_path = Path(args.batch_root) / "_figs_batch_index.json"
        with open(idx_path, "w") as f:
            json.dump(artifacts_list, f, indent=2)
        print(f"[build_graphics] Done. Wrote {idx_path}")


if __name__ == "__main__":
    main()
