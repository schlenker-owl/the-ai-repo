import numpy as np
from airoad.datasets.toy import make_classification_2d, standardize
from airoad.classic.linear.logreg_numpy import LogisticRegressionGD

def test_logreg_easy_data():
    X, y, *_ = make_classification_2d(n=200, margin=1.0, seed=7)
    X = standardize(X).X
    model = LogisticRegressionGD(lr=0.2, epochs=400, l2=0.0).fit(X, y)
    acc = model.accuracy(X, y)
    assert acc >= 0.9
