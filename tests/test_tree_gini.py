import numpy as np
from airoad.classic.tree.tree_gini import DecisionTreeGini

def make_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y

def test_tree_xor_high_acc():
    X, y = make_xor(n=300, seed=1)
    clf = DecisionTreeGini(max_depth=3, min_samples_split=2).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc >= 0.95
