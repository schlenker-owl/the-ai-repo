from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    n_components: int  # number of principal components to keep
    mean_: np.ndarray | None = None  # (d,)
    components_: np.ndarray | None = None  # (k, d)
    explained_variance_: np.ndarray | None = None  # (k,)
    explained_variance_ratio_: np.ndarray | None = None  # (k,)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        assert 1 <= self.n_components <= min(n, d), "Invalid n_components"

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # economy SVD, Xc = U S Vt
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # variances of PCs = S^2 / (n-1)
        var = (S * S) / max(n - 1, 1)
        total = var.sum()
        k = self.n_components
        self.components_ = Vt[:k, :]  # (k, d)
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / (total + 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_.T  # (n, k)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
