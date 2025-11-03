import numpy as np
from airoad.models.knn import KNNClassifier, KNNRegressor

def test_knn_classifier_acc():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(80, 2)) + np.array([+3, 0])
    X1 = rng.normal(size=(80, 2)) + np.array([-3, 0])
    X = np.vstack([X0, X1]); y = np.array([0]*80 + [1]*80)
    knn = KNNClassifier(k=5).fit(X, y)
    assert knn.accuracy(X, y) >= 0.9

def test_knn_regressor_mse():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(100, 1))
    y = (X[:, 0] ** 2) + 0.2 * rng.normal(size=100)
    knnr = KNNRegressor(k=7).fit(X, y)
    assert knnr.mse(X, y) <= 0.2
