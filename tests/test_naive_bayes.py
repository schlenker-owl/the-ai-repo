# tests/test_naive_bayes.py
import numpy as np

from airoad.models.naive_bayes import GaussianNB, MultinomialNB


def test_gaussian_nb_separable():
    rng = np.random.default_rng(0)
    X0 = rng.normal(-1.0, 0.5, size=(150, 3))
    X1 = rng.normal(+1.0, 0.5, size=(150, 3))
    X = np.vstack([X0, X1])
    y = np.array([0] * 150 + [1] * 150, dtype=np.int64)
    gnb = GaussianNB().fit(X, y)
    assert gnb.accuracy(X, y) >= 0.9


def test_multinomial_nb_toy():
    docs = ["spam spam ham", "ham eggs", "spam offer", "ham salad"]
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    # tiny vectorizer
    vocab = {}
    for d in docs:
        for w in d.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for i, d in enumerate(docs):
        for w in d.split():
            X[i, vocab[w]] += 1
    mnb = MultinomialNB(alpha=1.0).fit(X, y)
    assert mnb.accuracy(X, y) >= 0.9
