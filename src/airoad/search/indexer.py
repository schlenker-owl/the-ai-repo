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
class IndexConfig:
    index_type: str = "auto"  # "auto" | "faiss_flat" | "ivfpq" | "numpy"
    nlist: int = 1024  # IVF clusters (ivfpq)
    m: int = 64  # PQ sub-vector size
    nprobe: int = 16  # IVF probes


class VectorIndex:
    """FAISS index if available; else NumPy brute-force (cosine)."""

    def __init__(self, dim: int, cfg: IndexConfig):
        self.dim = dim
        self.cfg = cfg
        self.paths: List[str] = []
        self._faiss_index = None
        self._vecs = None  # (N,D) float32 normalized

    def build(self, vecs: np.ndarray, paths: List[str]):
        assert vecs.shape[0] == len(paths)
        self.paths = paths
        vecs = vecs.astype(np.float32)
        # L2 normalize as safety
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms

        use_faiss = HAS_FAISS and (self.cfg.index_type in ("auto", "faiss_flat", "ivfpq"))
        if use_faiss:
            if self.cfg.index_type in ("auto", "faiss_flat"):
                index = faiss.IndexFlatIP(self.dim)
                index.add(vecs)
                self._faiss_index = index
                return
            elif self.cfg.index_type == "ivfpq":
                quantizer = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFPQ(
                    quantizer, self.dim, self.cfg.nlist, self.cfg.m, 8
                )  # 8 bits per code
                index.train(vecs)
                index.add(vecs)
                index.nprobe = self.cfg.nprobe
                self._faiss_index = index
                return

        # fallback NumPy
        self._vecs = vecs

    def search(self, q: np.ndarray, topk: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        if self._faiss_index is not None:
            D, Index = self._faiss_index.search(q, topk)
            return D, Index
        # NumPy cosine similarity
        sims = q @ self._vecs.T  # (B,N)
        Index = np.argsort(-sims, axis=1)[:, :topk]
        D = np.take_along_axis(sims, Index, axis=1)
        return D, Index

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "paths": self.paths,
            "cfg": vars(self.cfg),
            "backend": "faiss" if self._faiss_index is not None else "numpy",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(out_dir / "index.faiss"))
        else:
            np.save(out_dir / "embeddings.npy", self._vecs)

    @staticmethod
    def load(in_dir: Path) -> "VectorIndex":
        meta = json.loads((in_dir / "meta.json").read_text())
        dim = int(meta["dim"])
        cfg = IndexConfig(**meta.get("cfg", {}))
        vi = VectorIndex(dim, cfg)
        vi.paths = meta["paths"]
        if meta.get("backend") == "faiss" and HAS_FAISS:
            vi._faiss_index = faiss.read_index(str(in_dir / "index.faiss"))
        else:
            vi._vecs = np.load(in_dir / "embeddings.npy").astype(np.float32)
        return vi
