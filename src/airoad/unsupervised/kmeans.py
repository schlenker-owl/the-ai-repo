# src/airoad/unsupervised/kmeans.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distances between every row in X and every center in C.
    X: (n, d), C: (k, d)  ->  D: (n, k)
    """
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    x2 = (X * X).sum(axis=1, keepdims=True)          # (n,1)
    c2 = (C * C).sum(axis=1, keepdims=True).T        # (1,k)
    return x2 + c2 - 2.0 * (X @ C.T)


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ initialization.
    """
    n, d = X.shape
    C = np.empty((k, d), dtype=np.float64)

    # 1) Pick first center uniformly
    i0 = rng.integers(0, n)
    C[0] = X[i0]

    # 2) Subsequent centers proportional to squared distance to nearest chosen center
    D = _pairwise_sq_dists(X, C[0:1]).ravel()  # distances to first center
    for j in range(1, k):
        total = D.sum()
        if not np.isfinite(total) or total <= 0.0:
            # Degenerate case (duplicates, all points equal to a center) — pick random
            idx = rng.integers(0, n)
        else:
            probs = D / total
            idx = rng.choice(n, p=probs)
        C[j] = X[idx]
        # update D to nearest chosen center so far
        D = np.minimum(D, _pairwise_sq_dists(X, C[j:j + 1]).ravel())
    return C


def _lloyd_iterations(
    X: np.ndarray,
    C_init: np.ndarray,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run Lloyd's algorithm starting from C_init.
    Returns (centers, labels, inertia).
    """
    n, d = X.shape
    k = C_init.shape[0]
    C = C_init.copy()
    labels = None

    for _ in range(max_iter):
        # E-step: assign to nearest center
        D = _pairwise_sq_dists(X, C)            # (n,k)
        new_labels = D.argmin(axis=1)           # (n,)

        # M-step: recompute centers (with empty-cluster handling)
        C_new = np.empty_like(C)
        Dmin = D.min(axis=1)                    # (n,) used for reseeding
        for j in range(k):
            mask = (new_labels == j)
            if mask.any():
                C_new[j] = X[mask].mean(axis=0)
            else:
                # Re-seed empty cluster to the globally farthest point from ANY center
                far_idx = int(np.argmax(Dmin))
                C_new[j] = X[far_idx]

        shift = np.linalg.norm(C_new - C)
        C = C_new
        if labels is not None and np.array_equal(new_labels, labels):
            # assignments stabilized; also check center shift for safety
            if shift < tol:
                break
        labels = new_labels

        if shift < tol:
            break

    inertia = float((_pairwise_sq_dists(X, C).min(axis=1)).sum())
    return C, labels, inertia


@dataclass
class KMeans:
    n_clusters: int
    max_iter: int = 100
    tol: float = 1e-4
    init: str = "kmeans++"     # "kmeans++" or "random"
    n_init: int = 5            # multiple restarts; keep best by inertia
    random_state: int = 0

    cluster_centers_: np.ndarray | None = None   # (k, d)
    inertia_: float | None = None

    def fit(self, X: np.ndarray):
        """
        Fit K-Means with multiple restarts to avoid poor local minima.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = int(self.n_clusters)
        assert k >= 1 and k <= n, "n_clusters must be in [1, n_samples]"

        best_inertia = np.inf
        best_C = None
        best_labels = None

        base_rng = np.random.default_rng(self.random_state)

        for i in range(self.n_init):
            # derive a deterministic sub-seed for reproducibility across restarts
            rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))

            if self.init == "kmeans++":
                C0 = _kmeans_plus_plus_init(X, k, rng)
            else:
                idx = rng.choice(n, size=k, replace=False)
                C0 = X[idx].copy()

            C, labels, inertia = _lloyd_iterations(
                X, C0, max_iter=self.max_iter, tol=self.tol, rng=rng
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_C = C
                best_labels = labels

        self.cluster_centers_ = best_C
        self.inertia_ = float(best_inertia)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.cluster_centers_ is not None, "Call fit() first."
        D = _pairwise_sq_dists(np.asarray(X, dtype=np.float64), self.cluster_centers_)
        return D.argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)
