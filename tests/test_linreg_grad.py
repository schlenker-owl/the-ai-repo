import numpy as np


def numerical_grad(X, y, W, eps=1e-6):
    # L(W) = mean((XW - y)^2)
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        Wp = W.copy()
        Wp[i, 0] += eps
        Wm = W.copy()
        Wm[i, 0] -= eps
        Lp = np.mean((X @ Wp - y) ** 2)
        Lm = np.mean((X @ Wm - y) ** 2)
        grad[i, 0] = (Lp - Lm) / (2 * eps)  # <-- no extra *2
    return grad


def test_linreg_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    X = np.c_[X, np.ones((X.shape[0], 1))]  # bias in last col
    w_true = rng.normal(size=(4, 1))
    y = X @ w_true + 0.01 * rng.normal(size=(20, 1))
    W = np.zeros((4, 1))
    grad_analytic = (2.0 / X.shape[0]) * (X.T @ (X @ W - y))
    grad_numeric = numerical_grad(X, y, W)
    assert np.allclose(grad_analytic, grad_numeric, atol=1e-5)
