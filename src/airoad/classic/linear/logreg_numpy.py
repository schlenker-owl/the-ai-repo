from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones((X.shape[0], 1))]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[neg])
    out[neg] = expz / (1.0 + expz)
    return out


@dataclass
class LogisticRegressionGD:
    lr: float = 0.1
    epochs: int = 800
    fit_intercept: bool = True
    l2: float = 0.0
    w: np.ndarray | None = None
    b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n, d = X.shape
        W = np.zeros((d + (0 if not self.fit_intercept else 1), 1), dtype=np.float64)
        X_ext = _add_bias(X) if self.fit_intercept else X

        for _ in range(self.epochs):
            logits = X_ext @ W
            p = _sigmoid(logits)
            # gradient of BCE with L2 on weights (not bias)
            err = p - y  # (n,1)
            grad = (X_ext.T @ err) / n
            if self.fit_intercept:
                grad[:-1, :] += (self.l2 / n) * W[:-1, :]
            else:
                grad += (self.l2 / n) * W
            W -= self.lr * grad

        if self.fit_intercept:
            self.w = W[:-1, 0]
            self.b = float(W[-1, 0])
        else:
            self.w = W[:, 0]
            self.b = 0.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.w is not None
        logits = X @ self.w + (self.b if self.fit_intercept else 0.0)
        return _sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(np.int32)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        return float((y_hat == y).mean())
