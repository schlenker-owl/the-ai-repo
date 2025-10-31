from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class StandardizeResult:
    X: np.ndarray
    mean: np.ndarray
    std: np.ndarray

def standardize(X: np.ndarray, eps: float = 1e-8) -> StandardizeResult:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + eps
    return StandardizeResult((X - mean) / std, mean.squeeze(0), std.squeeze(0))

def make_linear_regression(n: int = 200, d: int = 1, noise: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d, 1))
    b_true = rng.normal(size=(1,))
    y = X @ w_true + b_true + noise * rng.normal(size=(n, 1))
    return X.astype(np.float64), y.squeeze(1).astype(np.float64), w_true.squeeze(1), float(b_true)

def make_classification_2d(n: int = 300, margin: float = 0.5, seed: int = 42, flip_prob: float = 0.0):
    """
    Generate a *linearly separable* 2D dataset whose separation increases with `margin`.
    We first compute logits = X·w + b, then push each point away from the boundary by adding
    `margin * sign(logits)` before thresholding. Optionally flip a fraction of labels.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    w_true = rng.normal(size=(2,))
    # normalize w so 'margin' has consistent meaning
    w_true = w_true / (np.linalg.norm(w_true) + 1e-12)
    b_true = rng.normal()

    logits = X @ w_true + b_true
    logits = logits + margin * np.sign(logits)  # push away from boundary
    y = (logits > 0.0).astype(np.int32)

    if flip_prob > 0.0:
        mask = rng.random(n) < flip_prob
        y[mask] = 1 - y[mask]

    return X.astype(np.float64), y, w_true, float(b_true)
