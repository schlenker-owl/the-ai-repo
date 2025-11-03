import numpy as np
from airoad.unsupervised.gmm import GMM, normalized_mutual_info

def make_blobs(n=300, k=3, d=2, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * sep
    n_per = n // k
    Xs, ys = [], []
    for j in range(k):
        Xs.append(centers[j] + rng.normal(size=(n_per, d)))
        ys.append(np.full(n_per, j))
    return np.vstack(Xs), np.hstack(ys)

def test_gmm_nmi_high():
    X, y = make_blobs()
    gmm = GMM(n_components=3, max_iter=50, tol=1e-4, random_state=0).fit(X)
    pred = gmm.predict(X)
    nmi = normalized_mutual_info(pred, y)
    assert nmi >= 0.9
