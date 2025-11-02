# src/airoad/models/softmax_numpy.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((y.size, num_classes), dtype=np.float64)
    Y[np.arange(y.size), y] = 1.0
    return Y

def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

@dataclass
class SoftmaxRegressionGD:
    lr: float = 0.1
    epochs: int = 500
    l2: float = 0.0
    fit_intercept: bool = True

    W: np.ndarray | None = None   # (d, K)
    b: np.ndarray | None = None   # (K,)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        K = int(y.max()) + 1
        Y = _one_hot(y, K)

        W = np.zeros((d, K))
        b = np.zeros((K,))

        for _ in range(self.epochs):
            logits = X @ W + (b if self.fit_intercept else 0.0)   # (n,K)
            P = _softmax(logits)                                  # (n,K)
            # gradients
            G = (X.T @ (P - Y)) / n + 2.0 * self.l2 * W           # (d,K)
            gb = (P - Y).mean(axis=0) if self.fit_intercept else 0.0
            # update
            W -= self.lr * G
            if self.fit_intercept:
                b -= self.lr * gb

        self.W, self.b = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.W + (self.b if self.fit_intercept else 0.0)
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((self.predict(X) == y).mean())
