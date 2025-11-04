# src/airoad/models/naive_bayes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianNB:
    eps: float = 1e-9
    class_prior_: np.ndarray | None = None  # (K,)
    theta_: np.ndarray | None = None  # (K,d) means
    var_: np.ndarray | None = None  # (K,d) variances

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        K = int(y.max()) + 1

        theta = np.zeros((K, d))
        var = np.zeros((K, d))
        prior = np.zeros((K,))
        for k in range(K):
            Xk = X[y == k]
            prior[k] = Xk.shape[0] / n
            if Xk.shape[0] == 0:
                theta[k] = 0.0
                var[k] = 1.0
            else:
                theta[k] = Xk.mean(axis=0)
                var[k] = Xk.var(axis=0) + self.eps
        self.class_prior_, self.theta_, self.var_ = prior, theta, var
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        K, d = self.theta_.shape
        jll = np.zeros((X.shape[0], K))
        for k in range(K):
            # log N(x|mu, var) = -0.5 * [sum log(2π var) + sum (x-mu)^2/var]
            logp = -0.5 * (
                np.log(2.0 * np.pi * self.var_[k]).sum()
                + ((X - self.theta_[k]) ** 2 / self.var_[k]).sum(axis=1)
            )
            jll[:, k] = np.log(self.class_prior_[k] + 1e-12) + logp
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._joint_log_likelihood(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((self.predict(X) == y).mean())


@dataclass
class MultinomialNB:
    alpha: float = 1.0  # Laplace smoothing
    class_log_prior_: np.ndarray | None = None  # (K,)
    feature_log_prob_: np.ndarray | None = None  # (K,d)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        K = int(y.max()) + 1

        class_count = np.bincount(y, minlength=K).astype(np.float64)
        self.class_log_prior_ = np.log(class_count / n + 1e-12)

        # per-class feature counts with smoothing
        feature_count = np.zeros((K, d), dtype=np.float64)
        for k in range(K):
            feature_count[k] = X[y == k].sum(axis=0)
        smoothed = feature_count + self.alpha
        norm = smoothed.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed / (norm + 1e-12))
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_log_proba(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((self.predict(X) == y).mean())
