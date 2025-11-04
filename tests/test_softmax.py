# tests/test_softmax.py
import numpy as np

from airoad.models.softmax_numpy import SoftmaxRegressionGD


def _make(n=300, k=3, d=4, margin=0.9, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, k))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    b = rng.normal(size=(k,))
    X = rng.normal(size=(n, d))
    logits = X @ W + b + margin * np.sign(X @ W + b)
    y = logits.argmax(axis=1).astype(np.int64)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y


def test_softmax_acc():
    X, y = _make()
    clf = SoftmaxRegressionGD(lr=0.1, epochs=400, l2=1e-4).fit(X, y)
    assert clf.accuracy(X, y) >= 0.9
