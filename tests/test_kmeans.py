import numpy as np

from airoad.unsupervised.kmeans import KMeans


def make_blobs(n=300, k=3, d=2, sep=5.0, seed=1):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * sep
    n_per = n // k
    Xs, ys = [], []
    for j in range(k):
        Xs.append(centers[j] + rng.normal(size=(n_per, d)))
        ys.append(np.full(n_per, j))
    return np.vstack(Xs), np.hstack(ys)


def purity(y_true, y_pred):
    acc = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.any():
            vals, counts = np.unique(y_true[mask], return_counts=True)
            acc += counts.max()
    return float(acc / len(y_true))


def test_kmeans_purity_on_blobs():
    X, y = make_blobs(n=300, k=3, d=2, sep=6.0, seed=3)
    km = KMeans(n_clusters=3, random_state=0, max_iter=60).fit(X)
    p = purity(y, km.predict(X))
    assert p >= 0.9
