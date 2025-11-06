from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .backbone import BackboneConfig, BaseBackbone, build_backbone


@dataclass
class EmbedConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)  # <-- important fix
    batch_size: int = 64
    frame_stride: int = 5
    per_track_max_samples: int = 12
    min_box_size: int = 16
    class_filter: Optional[List[int]] = None  # e.g., [0] for person only


def _gather_needed_frames(track_df: pd.DataFrame, stride: int, kmax: int) -> Dict[int, List[int]]:
    need: Dict[int, List[int]] = {}
    for tid, g in track_df.groupby("track_id"):
        frames = g["frame"].astype(int).sort_values().tolist()
        frames = frames[:: max(1, stride)]
        if len(frames) > kmax:
            step = max(1, len(frames) // kmax)
            frames = frames[::step][:kmax]
        need[tid] = frames
    return need


def _read_video(vpath: Path):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vpath}")
    return cap


def _crop_from_frame(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2, ::-1]  # BGR->RGB
    return Image.fromarray(crop)


@torch.no_grad()
def embed_tracks_for_video(video_dir: Path, cfg: EmbedConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute per-track embeddings by sampling crops from analysis.mp4 using boxes from tracks.csv.
    Returns (meta_df, emb_array)
      meta_df columns: ["video","track_id","class_id","class_name","n_samples","f_min","f_max"]
      emb_array shape: (num_tracks, D) l2-normalized
    """
    tracks_csv = video_dir / "tracks.csv"
    analysis_mp4 = video_dir / "analysis.mp4"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing tracks.csv in {video_dir}")
    if not analysis_mp4.exists():
        raise FileNotFoundError(f"Missing analysis.mp4 in {video_dir}")

    tdf = pd.read_csv(tracks_csv)
    required_cols = {"frame", "track_id", "class_id", "class_name", "x1", "y1", "x2", "y2"}
    if not required_cols.issubset(set(tdf.columns)):
        raise ValueError(f"tracks.csv missing columns. Need at least {required_cols}")

    if cfg.class_filter is not None:
        tdf = tdf[tdf["class_id"].isin(cfg.class_filter)]

    need_frames = _gather_needed_frames(tdf, cfg.frame_stride, cfg.per_track_max_samples)

    by_frame: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    for _, row in tdf.iterrows():
        f = int(row["frame"])
        tid = int(row["track_id"])
        if tid not in need_frames or f not in need_frames[tid]:
            continue
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        by_frame.setdefault(f, []).append((tid, x1, y1, x2, y2))

    bb: BaseBackbone = build_backbone(cfg.backbone)

    cap = _read_video(analysis_mp4)
    feat_store: Dict[int, List[torch.Tensor]] = {}
    fmin: Dict[int, int] = {}
    fmax: Dict[int, int] = {}

    batch_imgs: List[torch.Tensor] = []
    batch_tids: List[int] = []

    cur_frame_idx = -1
    ok, frame = cap.read()
    while ok:
        cur_frame_idx += 1
        if cur_frame_idx in by_frame:
            for tid, x1, y1, x2, y2 in by_frame[cur_frame_idx]:
                w = max(1, x2 - x1 + 1)
                h = max(1, y2 - y1 + 1)
                if w < cfg.min_box_size or h < cfg.min_box_size:
                    continue
                crop_pil = _crop_from_frame(frame, (x1, y1, x2, y2))
                if crop_pil is None:
                    continue
                img_t = bb.preprocess(crop_pil).unsqueeze(0)  # 1CHW
                batch_imgs.append(img_t)
                batch_tids.append(tid)
                fmin[tid] = min(fmin.get(tid, cur_frame_idx), cur_frame_idx)
                fmax[tid] = max(fmax.get(tid, -1), cur_frame_idx)

            if len(batch_imgs) >= cfg.batch_size:
                batch = torch.cat(batch_imgs, dim=0).to(device=bb.device)
                feats = bb.encode_images(batch)  # (B,D)
                for t, fv in zip(batch_tids, feats):
                    feat_store.setdefault(t, []).append(fv.detach().cpu())
                batch_imgs, batch_tids = [], []

        ok, frame = cap.read()

    cap.release()

    if batch_imgs:
        batch = torch.cat(batch_imgs, dim=0).to(device=bb.device)
        feats = bb.encode_images(batch)
        for t, fv in zip(batch_tids, feats):
            feat_store.setdefault(t, []).append(fv.detach().cpu())
        batch_imgs, batch_tids = [], []

    # class/name lookup
    class_lookup = (
        tdf.groupby("track_id")[["class_id", "class_name"]].agg(lambda s: s.iloc[0]).reset_index()
    )
    class_map = {
        int(r.track_id): (int(r.class_id), str(r.class_name)) for _, r in class_lookup.iterrows()
    }

    tids = sorted(feat_store.keys())
    if not tids:
        meta = pd.DataFrame(
            columns=["video", "track_id", "class_id", "class_name", "n_samples", "f_min", "f_max"]
        )
        return meta, np.zeros((0, 512), dtype=np.float32)

    agg_feats = []
    meta_rows = []
    for tid in tids:
        v = torch.stack(feat_store[tid], dim=0)  # (N,D)
        mean = torch.nn.functional.normalize(v.mean(dim=0), p=2, dim=-1)
        agg_feats.append(mean.numpy().astype(np.float32))
        c_id, c_name = class_map.get(tid, (-1, "unknown"))
        meta_rows.append(
            {
                "video": video_dir.name,
                "track_id": int(tid),
                "class_id": int(c_id),
                "class_name": str(c_name),
                "n_samples": int(v.shape[0]),
                "f_min": int(fmin.get(tid, -1)),
                "f_max": int(fmax.get(tid, -1)),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    emb = np.vstack(agg_feats)
    return meta_df, emb
