from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _log_gauss(x, mu, cov):
    d = x.shape[1]
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov) + 1e-12
    xc = x - mu
    quad = np.einsum("ni,ij,nj->n", xc, cov_inv, xc)
    return -0.5 * (quad + d * np.log(2 * np.pi) + np.log(det))

def _contingency(pred, true, k_pred, k_true):
    C = np.zeros((k_pred, k_true), dtype=np.int64)
    for i in range(len(true)):
        C[pred[i], true[i]] += 1
    return C

def normalized_mutual_info(pred, true):
    pred = np.asarray(pred, dtype=int)
    true = np.asarray(true, dtype=int)
    k_pred = pred.max() + 1
    k_true = true.max() + 1
    C = _contingency(pred, true, k_pred, k_true)
    n = C.sum()
    pi = C.sum(axis=1) / n
    pj = C.sum(axis=0) / n
    outer = np.outer(pi, pj) + 1e-12
    nz = C > 0
    I = (C[nz] / n * np.log((C[nz] / n) / outer[nz])).sum()
    Hx = -(pi * np.log(pi + 1e-12)).sum()
    Hy = -(pj * np.log(pj + 1e-12)).sum()
    return float(2 * I / (Hx + Hy + 1e-12))

@dataclass
class GMM:
    n_components: int
    max_iter: int = 100
    tol: float = 1e-4
    reg_covar: float = 1e-6
    random_state: int = 0

    # fitted
    weights_: np.ndarray | None = None
    means_: np.ndarray | None = None
    covs_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        # init means from random points
        means = X[rng.choice(n, size=self.n_components, replace=False)].copy()
        weights = np.ones(self.n_components) / self.n_components
        covs = np.array([np.cov(X.T) + self.reg_covar * np.eye(d) for _ in range(self.n_components)])

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: responsibilities
            log_probs = np.stack([_log_gauss(X, means[k], covs[k]) + np.log(weights[k] + 1e-12)
                                  for k in range(self.n_components)], axis=1)  # (n,K)
            # log-sum-exp
            m = log_probs.max(axis=1, keepdims=True)
            probs = np.exp(log_probs - m)
            resp = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)

            # M-step
            Nk = resp.sum(axis=0) + 1e-12
            weights = Nk / n
            means = (resp.T @ X) / Nk[:, None]
            covs_new = []
            for k in range(self.n_components):
                xc = X - means[k]
                Sk = (resp[:, k][:, None] * xc).T @ xc / Nk[k]
                Sk += self.reg_covar * np.eye(d)
                covs_new.append(Sk)
            covs = np.stack(covs_new, axis=0)

            # log-likelihood
            ll = float((m + np.log(probs.sum(axis=1, keepdims=True) + 1e-12)).sum())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.weights_, self.means_, self.covs_ = weights, means, covs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_probs = np.stack([_log_gauss(X, self.means_[k], self.covs_[k]) + np.log(self.weights_[k] + 1e-12)
                              for k in range(self.n_components)], axis=1)
        return log_probs.argmax(axis=1)
