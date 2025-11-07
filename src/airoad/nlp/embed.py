from __future__ import annotations

from typing import List

import numpy as np


class TextEmbedder:
    """
    Prefer sentence-transformers; fall back to TF-IDF dense vectors (normalized).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self._backend = "tfidf"
        self._model = None
        self._tfidf = None
        self._dim = 0
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self._model = SentenceTransformer(name, device=device or "cpu")
            self._backend = "sbert"
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._backend = "tfidf"

    def fit(self, texts: List[str]) -> None:
        if self._backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._tfidf = TfidfVectorizer(max_features=4096, stop_words="english")
            self._tfidf.fit(texts)
            self._dim = int(min(4096, len(self._tfidf.get_feature_names_out())))

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sbert":
            emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)
        # tfidf fallback -> dense normalized
        X = self._tfidf.transform(texts).astype(np.float32)  # type: ignore
        if hasattr(X, "toarray"):
            X = X.toarray()  # ok at 4k features for small corpora
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return (X / norms).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim
