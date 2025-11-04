from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones((X.shape[0], 1))]


@dataclass
class LinearSVM:
    lam: float = 1e-2  # L2 penalty on weights (not bias)
    lr: float = 0.1  # step size for subgradient
    steps: int = 500
    fit_intercept: bool = True

    W: np.ndarray | None = None  # (d(+1),1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Primal, full-batch subgradient (Pegasos-style but full batch for determinism).
        y can be {0,1} or {-1,1}. Internally we use {-1,1}. Bias is not penalized.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        y = np.where(y > 0, 1.0, -1.0)

        Xext = _add_bias(X) if self.fit_intercept else X
        n, d = Xext.shape
        W = np.zeros((d, 1))
        reg_mask = np.ones_like(W)
        if self.fit_intercept:
            reg_mask[-1, 0] = 0.0  # don't penalize bias

        for t in range(1, self.steps + 1):
            margins = y * (Xext @ W)  # (n,1)
            mis = (margins < 1.0).astype(np.float64)  # indicator
            # grad = lam*W - (1/n) * sum_{mis} y_i x_i
            grad = self.lam * (reg_mask * W) - (Xext.T @ (y * mis)) / n
            W -= self.lr * grad

        self.W = W
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            return (X @ self.W[:-1, 0] + float(self.W[-1, 0])).reshape(-1, 1)
        return (X @ self.W[:, 0]).reshape(-1, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_function(X).ravel()
        return (s >= 0.0).astype(np.int64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X).ravel()
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((y_pred == y).mean())
