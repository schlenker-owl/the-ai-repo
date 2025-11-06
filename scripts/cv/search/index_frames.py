#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml
from PIL import Image

from airoad.search.backbone import ImageTextBackbone, SearchBackboneConfig
from airoad.search.indexer import IndexConfig, VectorIndex


def _list_analysis_videos(root: Path) -> List[Path]:
    return sorted([p / "analysis.mp4" for p in root.glob("*") if (p / "analysis.mp4").exists()])


def _sample_frames(video_path: Path, stride: int = 30, max_frames: int = 200) -> List[Image.Image]:
    imgs = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    f = -1
    while len(imgs) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        if f % stride != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(rgb))
    cap.release()
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/cv", help="root with per-video folders")
    ap.add_argument("--stride", type=int, default=30, help="sample one frame every N")
    ap.add_argument("--max-per-video", type=int, default=200)
    ap.add_argument("--config", type=str, default="configs/cv/search.yaml")
    ap.add_argument("--out", type=str, default="outputs/cv/.index_frames")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    bcfg = cfg.get("backbone", {})
    sb = ImageTextBackbone(
        SearchBackboneConfig(
            name=bcfg.get("name", "auto"),
            device=bcfg.get("device", None),
            dtype=bcfg.get("dtype", "float32"),
            image_size=int(bcfg.get("image_size", 224)),
        )
    )

    videos = _list_analysis_videos(Path(args.root).resolve())
    if not videos:
        print("[search/index_frames] No analysis.mp4 files found.")
        return

    print(f"[search/index_frames] Found {len(videos)} videos.")

    paths = []
    vecs = []
    for vp in videos:
        frames = _sample_frames(vp, stride=args.stride, max_frames=args.max_per_video)
        if not frames:
            continue
        pre = [sb.preprocess(fr) for fr in frames]
        feats = sb.encode_images(torch_stack(pre)).cpu().numpy()
        vecs.append(feats)
        # encode pseudo-paths for results
        for i in range(len(frames)):
            paths.append(f"{vp}#frame_{i*args.stride}")

    if not vecs:
        print("[search/index_frames] No frames sampled.")
        return

    emb = np.vstack(vecs).astype(np.float32)

    icfg = cfg.get("index", {})
    vindex = VectorIndex(
        sb.out_dim,
        IndexConfig(
            index_type=icfg.get("type", "auto"),
            nlist=int(icfg.get("nlist", 1024)),
            m=int(icfg.get("m", 64)),
            nprobe=int(icfg.get("nprobe", 16)),
        ),
    )
    vindex.build(emb, paths)
    out_dir = Path(args.out).resolve()
    vindex.save(out_dir)
    print(f"[search/index_frames] Saved index to {out_dir}")


def torch_stack(tensors):
    import torch

    return torch.stack(tensors, dim=0)


if __name__ == "__main__":
    main()
