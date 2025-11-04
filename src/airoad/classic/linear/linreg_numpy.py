from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones((X.shape[0], 1))]


@dataclass
class LinearRegressionGD:
    lr: float = 0.1
    epochs: int = 500
    fit_intercept: bool = True
    w: np.ndarray | None = None  # (d,) weights
    b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n, d = X.shape
        W = np.zeros((d + (0 if not self.fit_intercept else 1), 1), dtype=np.float64)

        if self.fit_intercept:
            X_ext = _add_bias(X)
        else:
            X_ext = X

        for _ in range(self.epochs):
            # y_hat = XW
            y_hat = X_ext @ W
            # grad = 2/n * X^T (XW - y)
            grad = (2.0 / n) * (X_ext.T @ (y_hat - y))
            W -= self.lr * grad

        if self.fit_intercept:
            self.w = W[:-1, 0]
            self.b = float(W[-1, 0])
        else:
            self.w = W[:, 0]
            self.b = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.w is not None
        y_hat = X @ self.w
        if self.fit_intercept:
            y_hat = y_hat + self.b
        return y_hat

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        diff = y_hat - y
        return float((diff @ diff) / y.shape[0])
