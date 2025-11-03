import numpy as np
from airoad.models.random_forest import RandomForestClassifier

def make_blobs(n=300, k=3, d=2, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * sep
    n_per = n // k
    Xs, ys = [], []
    for j in range(k):
        Xs.append(centers[j] + rng.normal(size=(n_per, d)))
        ys.append(np.full(n_per, j))
    return np.vstack(Xs), np.hstack(ys)

def test_rf_high_acc():
    X, y = make_blobs()
    rf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0).fit(X, y)
    assert rf.accuracy(X, y) >= 0.9
