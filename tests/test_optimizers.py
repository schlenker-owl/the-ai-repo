import numpy as np

from airoad.optimizers.optimizers import SGD, Adam, logistic_loss_grad


def _toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    w = rng.normal(size=(4,))
    w = w / (np.linalg.norm(w) + 1e-12)
    b = rng.normal()
    logits = X @ w + b
    logits = logits + 0.5 * np.sign(logits)
    y = (logits > 0).astype(float)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y


def _run(opt, steps=200):
    X, y = _toy()
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    last = None
    for _ in range(steps):
        loss, gw, gb = logistic_loss_grad(X, y.reshape(-1, 1), w, b, l2=0.0)
        w, b = opt.step(w, b, gw, gb)
        last = loss
    return last


def test_adam_beats_sgd_on_toy():
    sgd = _run(SGD(lr=0.1))
    adam = _run(Adam(lr=0.02))
    assert adam <= sgd  # Adam should not be worse on this toy
