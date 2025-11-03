# src/airoad/models/ridge_lasso.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _add_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones to X."""
    return np.c_[X, np.ones((X.shape[0], 1))]


def ridge_closed_form(
    X: np.ndarray, y: np.ndarray, lam: float, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """
    Solve:  min_{w,b} (1/n) * ||X w + b - y||^2 + lam * ||w||^2
    (bias is NOT penalized).

    Closed-form via normal equations on the (optionally) bias-augmented design.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n, _ = X.shape

    Xext = _add_bias(X) if fit_intercept else X
    A = (Xext.T @ Xext) / n                           # (d(+1), d(+1))
    b = (Xext.T @ y) / n                              # (d(+1), 1)
    I = np.eye(A.shape[0])
    if fit_intercept:
        I[-1, -1] = 0.0                               # do not penalize bias
    w_ext = np.linalg.solve(A + lam * I, b)           # (d(+1), 1)

    if fit_intercept:
        w = w_ext[:-1, 0]
        bias = float(w_ext[-1, 0])
    else:
        w = w_ext[:, 0]
        bias = 0.0
    return w, bias


def ridge_gd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    lr: float = 0.1,
    epochs: int = 500,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Gradient descent for the ridge objective (bias unpenalized).

    Objective: (1/n) * ||X w + b - y||^2 + lam * ||w||^2
    Gradient (with bias column in Xext): (2/n) Xext^T (Xext W - y) + 2*lam*[w;0]
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n, _ = X.shape

    Xext = _add_bias(X) if fit_intercept else X
    W = np.zeros((Xext.shape[1], 1), dtype=np.float64)

    # Mask so the last coord (bias) is not penalized
    reg_mask = np.ones_like(W)
    if fit_intercept:
        reg_mask[-1, 0] = 0.0

    for _ in range(epochs):
        resid = Xext @ W - y
        grad = (2.0 / n) * (Xext.T @ resid) + 2.0 * lam * (reg_mask * W)
        W -= lr * grad

    if fit_intercept:
        return W[:-1, 0], float(W[-1, 0])
    return W[:, 0], 0.0


def soft_threshold(z: float, thr: float) -> float:
    """Soft-thresholding operator S(z, thr)."""
    if z > thr:
        return z - thr
    if z < -thr:
        return z + thr
    return 0.0


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    fit_intercept: bool = True,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> tuple[np.ndarray, float]:
    """
    Solve:  min_{w,b} (1/2n) * ||y - (X w + b)||^2 + lam * ||w||_1
    via cyclic coordinate descent. Intercept is unpenalized.

    Notes:
      * Assumes features are roughly standardized for best behavior.
      * We place the intercept as the last column of the augmented design.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n, _ = X.shape

    Xext = _add_bias(X) if fit_intercept else X            # (n, d(+1))
    m = Xext.shape[1]
    W = np.zeros((m, 1), dtype=np.float64)

    # Precompute column squared norms
    col_norm2 = (Xext ** 2).sum(axis=0, keepdims=True)     # (1, m)

    for _ in range(max_iter):
        W_prev = W.copy()

        # ---- Intercept update (last coordinate), unpenalized ----
        if fit_intercept:
            j = m - 1
            # residual excluding intercept contribution
            r = y - (Xext @ W) + Xext[:, [j]] * W[j, 0]
            numer = float(r.sum()) / n
            denom = float(col_norm2[0, j]) / n
            denom = max(denom, 1e-12)
            W[j, 0] = numer / denom

        # ---- L1-penalized weights (all but intercept) ----
        J = m - 1 if fit_intercept else m
        for j in range(J):
            # residual excluding j-th contribution
            r = y - (Xext @ W) + Xext[:, [j]] * W[j, 0]
            # rho = (1/n) * x_j^T r
            rho = float(((Xext[:, [j]].T @ r).item()) / n)
            # z = (1/n) * ||x_j||^2
            z = float(col_norm2[0, j]) / n
            z = max(z, 1e-12)
            W[j, 0] = soft_threshold(rho, lam) / z

        # Convergence check
        if np.linalg.norm(W - W_prev) < tol:
            break

    if fit_intercept:
        return W[:-1, 0], float(W[-1, 0])
    return W[:, 0], 0.0
