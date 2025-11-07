from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


@dataclass
class IndexCfg:
    index_type: str = "auto"  # "auto" | "faiss_flat" | "ivfpq" | "numpy"
    nlist: int = 1024
    m: int = 64
    nprobe: int = 16


class IndexStore:
    def __init__(self, dim: int, cfg: IndexCfg):
        self.dim = dim
        self.cfg = cfg
        self.paths: List[str] = []
        self._faiss = None
        self._vecs = None

    def build(self, vecs: np.ndarray, paths: List[str]):
        assert vecs.shape[0] == len(paths)
        self.paths = paths
        vecs = vecs.astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        if HAS_FAISS and self.cfg.index_type in ("auto", "faiss_flat", "ivfpq"):
            if self.cfg.index_type in ("auto", "faiss_flat"):
                idx = faiss.IndexFlatIP(self.dim)
                idx.add(vecs)
                self._faiss = idx
            else:
                q = faiss.IndexFlatIP(self.dim)
                idx = faiss.IndexIVFPQ(q, self.dim, self.cfg.nlist, self.cfg.m, 8)
                idx.train(vecs)
                idx.add(vecs)
                idx.nprobe = self.cfg.nprobe
                self._faiss = idx
            return
        self._vecs = vecs

    def search(self, q: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        if self._faiss is not None:
            distances, indices = self._faiss.search(q, topk)
            return distances, indices
        sims = q @ self._vecs.T
        indices = np.argsort(-sims, axis=1)[:, :topk]
        distances = np.take_along_axis(sims, indices, axis=1)
        return distances, indices

    def save(self, out_dir: Path | str):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "paths": self.paths,
            "cfg": {
                "index_type": self.cfg.index_type,
                "nlist": self.cfg.nlist,
                "m": self.cfg.m,
                "nprobe": self.cfg.nprobe,
            },
            "backend": "faiss" if self._faiss is not None else "numpy",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(out_dir / "index.faiss"))
        else:
            np.save(out_dir / "embeddings.npy", self._vecs)

    @staticmethod
    def load(in_dir: Path | str) -> "IndexStore":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        st = IndexStore(int(meta["dim"]), IndexCfg(**meta.get("cfg", {})))
        st.paths = meta["paths"]
        if meta.get("backend") == "faiss" and HAS_FAISS:
            st._faiss = faiss.read_index(str(in_dir / "index.faiss"))
        else:
            st._vecs = np.load(in_dir / "embeddings.npy").astype(np.float32)
        return st
