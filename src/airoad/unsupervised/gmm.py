from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

EPS = 1e-12


# ===========================
#  Stable helpers
# ===========================


def _logsumexp(a: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """Numerically stable logsumexp along an axis."""
    amax = np.max(a, axis=axis, keepdims=True)
    return amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True) + EPS)


# ===========================
#  Log-density helpers
# ===========================


def _log_gauss_full(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log N(x | mu, cov) per row (full covariance)."""
    d = x.shape[1]
    cov_reg = cov + 1e-12 * np.eye(d, dtype=x.dtype)
    cov_inv = np.linalg.inv(cov_reg)
    _, logdet = np.linalg.slogdet(cov_reg)
    logdet = float(logdet)
    xc = x - mu
    quad = np.einsum("ni,ij,nj->n", xc, cov_inv, xc)
    return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)


def _log_gauss_diag(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log N(x | mu, diag(var)) per row."""
    d = x.shape[1]
    var_reg = var + 1e-12
    inv = 1.0 / var_reg
    xc = x - mu
    quad = (xc * xc * inv).sum(axis=1)
    logdet = np.log(var_reg).sum()
    return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)


def _log_gauss_spherical(x: np.ndarray, mu: np.ndarray, sigma2: float) -> np.ndarray:
    """Log N(x | mu, sigma^2 I) per row."""
    d = x.shape[1]
    s2 = float(sigma2 + 1e-12)
    xc = x - mu
    quad = (xc * xc).sum(axis=1) / s2
    logdet = d * np.log(s2)
    return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)


# ===========================
#  NMI (test utility)
# ===========================


def _contingency(pred: np.ndarray, true: np.ndarray, k_pred: int, k_true: int) -> np.ndarray:
    C = np.zeros((k_pred, k_true), dtype=np.int64)
    for i in range(len(true)):
        C[pred[i], true[i]] += 1
    return C


def normalized_mutual_info(pred: np.ndarray, true: np.ndarray) -> float:
    """
    NMI = 2 * I(P;T) / (H(P) + H(T)), permutation-invariant.
    """
    pred = np.asarray(pred, dtype=int)
    true = np.asarray(true, dtype=int)
    kP, kT = pred.max() + 1, true.max() + 1
    C = _contingency(pred, true, kP, kT)
    n = C.sum()
    if n == 0:
        return 0.0
    p = C / n
    pP = p.sum(axis=1, keepdims=True)  # (kP,1)
    pT = p.sum(axis=0, keepdims=True)  # (1,kT)
    with np.errstate(divide="ignore", invalid="ignore"):
        logterm = np.where(p > 0, np.log(p / (pP @ pT)), 0.0)
    mutual_info = float((p * logterm).sum())
    H_P = float(-(pP[pP > 0] * np.log(pP[pP > 0])).sum())
    H_T = float(-(pT[pT > 0] * np.log(pT[pT > 0])).sum())
    return 0.0 if H_P + H_T == 0.0 else 2.0 * mutual_info / (H_P + H_T)


# ===========================
#  GMM with K-Means init
# ===========================


@dataclass
class GMM:
    n_components: int
    max_iter: int = 100
    tol: float = 1e-4
    reg_covar: float = 1e-8  # slightly smaller regularizer helps separation on simple blobs
    random_state: int = 0
    init: Literal["kmeans", "random"] = "kmeans"
    n_init: int = 10  # EM restarts; keep best ll
    covariance_type: Literal["full", "diag", "spherical"] = "full"
    # Enable whitening by default for more robust init/convergence on common datasets
    whiten: bool = True

    # learned (in whatever space we trained in: whitened or raw)
    weights_: np.ndarray | None = None  # (K,)
    means_: np.ndarray | None = None  # (K,D)
    covs_: np.ndarray | None = None  # (K,D,D) or (K,1,1)
    # whitening params (used only if whiten=True)
    x_mean_: np.ndarray | None = None
    x_std_: np.ndarray | None = None

    # ---------- initialization ----------

    def _kmeans_init(self, X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize from in-repo KMeans on the *same* space as EM (whitened if self.whiten).
        """
        from airoad.unsupervised.kmeans import KMeans

        n, d = X.shape
        km = KMeans(
            n_clusters=self.n_components,
            init="kmeans++",
            n_init=50,  # stronger restarts for robust centers
            max_iter=500,
            tol=1e-9,
            random_state=seed,
        ).fit(X)
        labels = km.predict(X)

        weights, means, covs = [], [], []
        for k in range(self.n_components):
            mask = labels == k
            cnt = int(mask.sum())
            if cnt == 0:
                # Guard: pick a random point if k-means produced an empty cluster
                rng = np.random.default_rng(seed + 101 + k)
                idx = int(rng.integers(0, n))
                mu = X[idx]
                cov = np.eye(d) if self.covariance_type != "spherical" else np.array([[1.0]])
                w = 1.0 / self.n_components
            else:
                Xk = X[mask]
                mu = Xk.mean(axis=0)
                xc = Xk - mu
                if self.covariance_type == "full":
                    Sk = (xc.T @ xc) / max(1, cnt) + self.reg_covar * np.eye(d)
                    cov = Sk
                elif self.covariance_type == "diag":
                    var = xc.var(axis=0) + self.reg_covar
                    cov = np.diag(var)
                else:
                    var = float(xc.var() + self.reg_covar)
                    cov = np.array([[var]])
                w = cnt / n
            weights.append(w)
            means.append(mu)
            covs.append(cov)
        return (
            np.asarray(weights, dtype=X.dtype),
            np.stack(means, axis=0),
            np.stack(covs, axis=0),
        )

    def _init_params(
        self, X: np.ndarray, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.init == "kmeans":
            w, m, c = self._kmeans_init(X, seed)
        else:
            n, d = X.shape
            rng = np.random.default_rng(seed)
            m = X[rng.choice(n, size=self.n_components, replace=False)].copy()
            if self.covariance_type == "full":
                c = np.array([np.eye(d) for _ in range(self.n_components)])
            elif self.covariance_type == "diag":
                c = np.array([np.eye(d) for _ in range(self.n_components)])
            else:
                c = np.array([[[1.0]] for _ in range(self.n_components)])
            w = np.ones(self.n_components, dtype=X.dtype) / self.n_components
        init_ll = self._loglik(X, w, m, c)
        return w, m, c, init_ll

    # ---------- log-likelihood ----------

    def _component_logps(
        self, X: np.ndarray, w: np.ndarray, m: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        if self.covariance_type == "full":
            return np.stack(
                [
                    _log_gauss_full(X, m[k], c[k]) + np.log(w[k] + EPS)
                    for k in range(self.n_components)
                ],
                axis=1,
            )
        elif self.covariance_type == "diag":
            return np.stack(
                [
                    _log_gauss_diag(X, m[k], np.diag(c[k])) + np.log(w[k] + EPS)
                    for k in range(self.n_components)
                ],
                axis=1,
            )
        else:
            return np.stack(
                [
                    _log_gauss_spherical(X, m[k], float(c[k][0, 0])) + np.log(w[k] + EPS)
                    for k in range(self.n_components)
                ],
                axis=1,
            )

    def _loglik(self, X: np.ndarray, w: np.ndarray, m: np.ndarray, c: np.ndarray) -> float:
        logp = self._component_logps(X, w, m, c)
        return float(_logsumexp(logp, axis=1, keepdims=True).sum())

    # ---------- EM core ----------

    def _em_run(self, X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        w, m, c, init_ll = self._init_params(X, seed)
        prev_ll = init_ll

        # Track best params within this EM run in case of oscillation
        best_ll = prev_ll
        best_wmc = (w.copy(), m.copy(), c.copy())

        for _ in range(self.max_iter):
            # E-step
            logp = self._component_logps(X, w, m, c)
            mlog = np.max(logp, axis=1, keepdims=True)
            resp_unnorm = np.exp(logp - mlog)
            r = resp_unnorm / (resp_unnorm.sum(axis=1, keepdims=True) + EPS)  # responsibilities

            # M-step
            Nk = r.sum(axis=0) + EPS
            w = Nk / X.shape[0]
            m = (r.T @ X) / Nk[:, None]

            if self.covariance_type == "full":
                c_list = []
                for k in range(self.n_components):
                    xc = X - m[k]
                    Sk = (r[:, k][:, None] * xc).T @ xc / Nk[k]
                    Sk += self.reg_covar * np.eye(X.shape[1])
                    c_list.append(Sk)
                c = np.stack(c_list, axis=0)
            elif self.covariance_type == "diag":
                c_list = []
                for k in range(self.n_components):
                    xc = X - m[k]
                    var = (r[:, k][:, None] * (xc * xc)).sum(axis=0) / Nk[k]
                    var = var + self.reg_covar
                    c_list.append(np.diag(var))
                c = np.stack(c_list, axis=0)
            else:
                d = X.shape[1]
                sig = []
                for k in range(self.n_components):
                    xc = X - m[k]
                    var = (r[:, k][:, None] * (xc * xc)).sum() / (Nk[k] * d)
                    sig.append([[float(var + self.reg_covar)]])
                c = np.array(sig)

            # LL + early stop
            curr_ll = self._loglik(X, w, m, c)
            if curr_ll > best_ll:
                best_ll = curr_ll
                best_wmc = (w.copy(), m.copy(), c.copy())

            if curr_ll - prev_ll < self.tol:
                break
            prev_ll = curr_ll

        final_w, final_m, final_c = best_wmc
        final_ll = best_ll

        # Likelihood guard: if degenerate or worse than init, re-init once with a different seed
        if not np.isfinite(final_ll) or final_ll + 1e-8 < init_ll:
            w2, m2, c2, ll2 = self._init_params(X, seed + 1337)
            if ll2 > final_ll:
                final_w, final_m, final_c, final_ll = w2, m2, c2, ll2

        return final_w, final_m, final_c, final_ll

    # ---------- API ----------

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if self.whiten:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-12
            X_use = (X - mean) / std
            self.x_mean_, self.x_std_ = mean, std
        else:
            X_use = X
            self.x_mean_, self.x_std_ = None, None

        best_ll = -np.inf
        best_params = None
        for run in range(self.n_init):
            seed = int(self.random_state + 9973 * run)
            w, m, c, ll = self._em_run(X_use, seed)
            if ll > best_ll:
                best_ll = ll
                best_params = (w, m, c)

        self.weights_, self.means_, self.covs_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_use = (X - self.x_mean_) / self.x_std_ if (self.x_mean_ is not None) else X
        logp = self._component_logps(X_use, self.weights_, self.means_, self.covs_)
        return logp.argmax(axis=1)
