#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from airoad.reid.matcher import MatchConfig, associate


def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/cv", help="Folder with per-video outputs")
    ap.add_argument("--config", type=str, default="configs/cv/reid.yaml")
    args = ap.parse_args()

    cfg_d = _load_yaml(Path(args.config))
    mcfg = MatchConfig(
        min_sim=float(cfg_d.get("associate", {}).get("min_sim", 0.45)),
        same_class_only=bool(cfg_d.get("associate", {}).get("same_class_only", True)),
        max_same_video_time_overlap=int(
            cfg_d.get("associate", {}).get("max_same_video_time_overlap", 0)
        ),
        allow_merge_same_video=bool(
            cfg_d.get("associate", {}).get("allow_merge_same_video", False)
        ),
    )

    emb_dir = Path(args.root).resolve() / "reid" / "embeddings"
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings dir not found: {emb_dir}. Run build_track_embeddings.py first."
        )

    meta_files = sorted(emb_dir.glob("*_track_meta.csv"))
    emb_files = sorted(emb_dir.glob("*_track_embeds.npz"))
    if not meta_files or not emb_files or len(meta_files) != len(emb_files):
        raise RuntimeError("Missing or mismatched embeddings/meta files.")

    metas = [pd.read_csv(mf) for mf in meta_files]
    embs = [np.load(ef)["emb"] for ef in emb_files]

    print(
        f"[reid] Associating {sum(m.shape[0] for m in metas)} tracks across {len(metas)} videos..."
    )
    global_map = associate(metas, embs, mcfg)  # adds 'global_id'

    # Write outputs
    out_dir = Path(args.root).resolve() / "reid"
    out_dir.mkdir(parents=True, exist_ok=True)

    map_csv = out_dir / "global_id_map.csv"
    global_map.to_csv(map_csv, index=False)

    # merged analytics
    analytic = {
        "num_videos": len(metas),
        "num_tracks": int(global_map.shape[0]),
        "num_globals": int(global_map["global_id"].nunique()),
        "unique_by_class": global_map.groupby("class_name")["global_id"].nunique().to_dict(),
        "tracks_by_video": global_map.groupby("video")["track_id"].nunique().to_dict(),
    }
    with open(out_dir / "merged_analytics.json", "w") as f:
        json.dump(analytic, f, indent=2)

    print(f"[reid] Wrote {map_csv} and merged_analytics.json")


if __name__ == "__main__":
    main()
