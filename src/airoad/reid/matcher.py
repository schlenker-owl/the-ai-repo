from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchConfig:
    min_sim: float = 0.45  # cosine
    same_class_only: bool = True
    # gating inside same video: don't merge tracks that overlap heavily in time
    max_same_video_time_overlap: int = 0  # frames; 0 => disallow any overlap
    # within-video merges allowed? (often False for MOT outputs)
    allow_merge_same_video: bool = False


def _time_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    a0, a1 = a
    b0, b1 = b
    if a1 < b0 or b1 < a0:
        return 0
    return min(a1, b1) - max(a0, b0) + 1


def associate(
    metas: List[pd.DataFrame],
    embs: List[np.ndarray],
    cfg: MatchConfig,
) -> pd.DataFrame:
    """
    Inputs: per-video meta DataFrames (columns: video, track_id, class_id, class_name, f_min, f_max)
            and embeddings arrays (Ti, D) for each video in the same order.
    Output: DataFrame mapping rows -> global_id with merged analytics.
    """
    # concat
    meta = pd.concat(metas, ignore_index=True)
    emb = np.vstack(embs).astype(np.float32)  # (N,D)
    N = emb.shape[0]

    # build similarity
    sim = cosine_similarity(emb)  # (N,N)
    np.fill_diagonal(sim, -1.0)

    # constraints
    vids = meta["video"].tolist()
    cids = meta["class_id"].astype(int).tolist()
    ranges = list(zip(meta["f_min"].astype(int).tolist(), meta["f_max"].astype(int).tolist()))

    # adjacency
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i, j] < cfg.min_sim:
                continue
            # class constraint
            if cfg.same_class_only and cids[i] != cids[j]:
                continue
            # same video gating
            if vids[i] == vids[j]:
                if not cfg.allow_merge_same_video:
                    continue
                ov = _time_overlap(ranges[i], ranges[j])
                if ov > cfg.max_same_video_time_overlap:
                    continue
            # link both ways
            adj[i].append(j)
            adj[j].append(i)

    # connected components -> global_id
    visited = [False] * N
    gid = [-1] * N
    gid_counter = 0
    for i in range(N):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        gid[i] = gid_counter
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    gid[v] = gid_counter
                    queue.append(v)
        gid_counter += 1

    meta_out = meta.copy()
    meta_out["global_id"] = gid
    return meta_out
