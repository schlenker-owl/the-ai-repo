import numpy as np

from airoad.unsupervised.pca import PCA


def test_pca_variance_and_reconstruction():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.normal(size=(n, d))
    # make correlated
    A = rng.normal(size=(d, d))
    X = X @ np.linalg.cholesky(A @ A.T)
    pca = PCA(n_components=2).fit(X)
    assert pca.explained_variance_ratio_.sum() > 0.5  # top-2 capture decent variance

    Z = pca.transform(X)
    Xr = pca.inverse_transform(Z)
    rec_mse = float(np.mean((X - Xr) ** 2))
    assert rec_mse < np.var(X) * 0.6  # crude but fast gate
