import numpy as np
from airoad.classic.linear.ridge_lasso import ridge_closed_form, ridge_gd, lasso_coordinate_descent

def test_ridge_closed_form_vs_gd():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    w_true = rng.normal(size=(5,)); b_true = rng.normal()
    y = X @ w_true + b_true + 0.1 * rng.normal(size=100)
    w_cf, b_cf = ridge_closed_form(X, y, lam=0.3, fit_intercept=True)
    w_gd, b_gd = ridge_gd(X, y, lam=0.3, lr=0.1, epochs=400, fit_intercept=True)
    assert np.allclose(w_cf, w_gd, atol=2e-3)
    assert abs(b_cf - b_gd) < 2e-3

def test_lasso_shrinkage_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    w_true = np.zeros(6); w_true[:2] = 2.0
    y = X @ w_true + 0.1 * rng.normal(size=80)
    zeros = []
    for lam in [0.0, 0.1, 0.3, 0.7]:
        w, b = lasso_coordinate_descent(X, y, lam, fit_intercept=True, max_iter=1500)
        zeros.append(int((np.abs(w) < 1e-8).sum()))
    # non-decreasing number of zeros as λ increases
    assert zeros == sorted(zeros)
