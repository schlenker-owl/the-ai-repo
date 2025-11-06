#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from airoad.reid.backbone import BackboneConfig
from airoad.reid.embedder import EmbedConfig, embed_tracks_for_video


def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/cv", help="Folder with per-video outputs")
    ap.add_argument("--config", type=str, default="configs/cv/reid.yaml")
    args = ap.parse_args()

    cfg_d = _load_yaml(Path(args.config))
    embed_cfg = EmbedConfig(
        backbone=BackboneConfig(
            name=str(cfg_d.get("backbone", {}).get("name", "auto")),
            device=cfg_d.get("backbone", {}).get("device", None),
            dtype=str(cfg_d.get("backbone", {}).get("dtype", "float32")),
            image_size=int(cfg_d.get("backbone", {}).get("image_size", 224)),
        ),
        batch_size=int(cfg_d.get("embed", {}).get("batch_size", 64)),
        frame_stride=int(cfg_d.get("embed", {}).get("frame_stride", 5)),
        per_track_max_samples=int(cfg_d.get("embed", {}).get("per_track_max_samples", 12)),
        min_box_size=int(cfg_d.get("embed", {}).get("min_box_size", 16)),
        class_filter=cfg_d.get("embed", {}).get("class_filter", None),
    )

    root = Path(args.root).resolve()
    out_root = root / "reid" / "embeddings"
    out_root.mkdir(parents=True, exist_ok=True)

    video_dirs = [p for p in root.glob("*") if p.is_dir() and (p / "tracks.csv").exists()]
    if not video_dirs:
        print(f"[reid] No video folders with tracks.csv found under {root}")
        return

    print(f"[reid] Found {len(video_dirs)} video folders")

    for vdir in video_dirs:
        print(f"[reid] Embedding tracks for {vdir.name} ...")
        meta_df, emb = embed_tracks_for_video(vdir, embed_cfg)
        npz_path = out_root / f"{vdir.name}_track_embeds.npz"
        csv_path = out_root / f"{vdir.name}_track_meta.csv"
        # save
        np.savez_compressed(npz_path, emb=emb.astype(np.float32))
        meta_df.to_csv(csv_path, index=False)
        print(f"[reid] Wrote {npz_path.name} and {csv_path.name}")


if __name__ == "__main__":
    main()
