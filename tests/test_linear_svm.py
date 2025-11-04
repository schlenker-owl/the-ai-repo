import numpy as np

from airoad.classic.linear.linear_svm import LinearSVM


def make_linear_sep(n=300, d=4, margin=0.6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d,))
    w = w / (np.linalg.norm(w) + 1e-12)
    b = rng.normal()
    logits = X @ w + b
    logits = logits + margin * np.sign(logits)
    y = (logits > 0).astype(int)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y


def test_svm_linearly_separable():
    X, y = make_linear_sep(n=300, d=5, margin=0.8, seed=3)
    svm = LinearSVM(lam=1e-2, lr=0.1, steps=400).fit(X, y)
    assert svm.accuracy(X, y) >= 0.9
