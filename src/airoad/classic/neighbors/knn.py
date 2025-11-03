from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A * A).sum(axis=1, keepdims=True)
    B2 = (B * B).sum(axis=1, keepdims=True).T
    return A2 + B2 - 2 * (A @ B.T)

@dataclass
class KNNClassifier:
    k: int = 5

    # fit
    X_: np.ndarray | None = None
    y_: np.ndarray | None = None
    n_classes_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.int64).ravel()
        self.n_classes_ = int(self.y_.max()) + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        D = _pairwise_sq_dists(X, self.X_)  # (n_test, n_train)
        idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=np.int64)
        for i in range(X.shape[0]):
            for lab in self.y_[idx[i]]:
                votes[i, lab] += 1
        return votes.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((self.predict(X) == y).mean())

@dataclass
class KNNRegressor:
    k: int = 5

    X_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64).ravel()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        D = _pairwise_sq_dists(X, self.X_)
        idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]
        return self.y_[idx].mean(axis=1)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        diff = pred - y.ravel()
        return float((diff @ diff) / len(diff))
