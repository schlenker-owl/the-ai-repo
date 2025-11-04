# Unsupervised Learning — PCA · K-Means · GMM (v0.2)

This guide explains the **unsupervised** modules in **the-ai-repo** (`airoad`) with: intuition → math (GitHub-friendly) → implementation notes → quick commands → tiny acceptance checks.

> **Math on GitHub:** inline math uses `$…$`; display equations use `$$…$$` (we use fenced blocks with `math` below).

---

## Contents

1. [Environment](#environment)
2. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [K-Means Clustering](#k-means-clustering)
4. [Gaussian Mixture Models (GMM, EM)](#gaussian-mixture-models-gmm-em)
5. [PCA → K-Means Pipeline](#pca--k-means-pipeline)
6. [Tests & Acceptance Gates](#tests--acceptance-gates)
7. [Common Pitfalls & Debugging](#common-pitfalls--debugging)
8. [Next Steps](#next-steps)
9. [File Map (Unsupervised)](#file-map-unsupervised)

---

## Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pytest -q
````

---

## Principal Component Analysis (PCA)

**Paths**

* Code: `src/airoad/unsupervised/pca.py`
* Script: `scripts/run_pca.py`
* Test: `tests/test_pca.py`

### Intuition

PCA finds **orthogonal directions** (principal components) that **maximize variance**. Projecting onto the top-$k$ components gives a lower-dimensional representation that preserves as much variance as possible.

### Math (center → SVD → project)

1. **Center** data $X \in \mathbb{R}^{n \times d}$ by mean $\mu$:

```math
X_c \;=\; X \;-\; \mathbf{1}\,\mu^\top
```

2. **SVD** (economy):

```math
X_c \;=\; U\,S\,V^\top,
\qquad
\text{PC directions} = V,\quad
\text{PC variances} = \frac{S^2}{n-1}.
```

3. **Keep top-$k$** rows of $V^\top$ as $W_k \in \mathbb{R}^{d \times k}$ and **project**:

```math
Z \;=\; X_c\,W_k \;\in\; \mathbb{R}^{n \times k}.
```

4. **Reconstruct** (optional):

```math
\hat{X} \;=\; Z\,W_k^\top + \mu.
```

5. **Explained variance ratio (EVR)** for top-$k$:

```math
\text{EVR}_k \;=\; \frac{\sum_{i=1}^{k} S_i^2}{\sum_{i=1}^{\min(n,d)} S_i^2}.
```

### Implementation Notes

* Uses **SVD** (stable) rather than covariance eigen-decomposition.
* Stores: `mean_`, `components_ (k×d)`, `explained_variance_`, `explained_variance_ratio_`.
* API: `fit`, `transform`, `inverse_transform`, `fit_transform`.

### Quick Commands

```bash
# Run PCA on correlated synthetic data, show variance ratio & recon error
uv run python scripts/run_pca.py --n 400 --d 5 --k 2
# → PCA: k=2  variance_ratio_sum=0.8xx  recon_mse=...
```

### Learn-By-Tweaking

* Increase `k` → EVR increases, reconstruction MSE drops.
* Compare highly correlated vs. independent features; note EVR difference.
* Standardize features first if scales differ widely.

---

## K-Means Clustering

**Paths**

* Code: `src/airoad/unsupervised/kmeans.py`
* Script: `scripts/run_kmeans.py`
* Test: `tests/test_kmeans.py`

### Intuition

Partition $n$ points into $k$ clusters by minimizing **within-cluster squared distance** (distortion). Alternate **assignment** (nearest center) and **update** (mean of cluster points).

### Objective

Given centers $C={c_1,\dots,c_k}$ and labels $\ell_i \in {1,\dots,k}$,

```math
\min_{C,\;\ell}\;\sum_{i=1}^{n} \big\lVert x_i - c_{\ell_i} \big\rVert_2^2.
```

The optimal objective value is often called **inertia**.

### Algorithm (k-means++)

1. **Init** with **k-means++** (diverse seeds; fewer bad local minima).
2. **Lloyd iterations** until convergence:

   * Assign each point to the nearest center.
   * Update each center to the **mean** of its assigned points.
   * Handle empty clusters by reseeding (globally farthest point).
3. Run **multiple restarts** (`n_init`) and keep the best inertia.

Useful identity for fast distances:

```math
\lVert x - c \rVert_2^2 \;=\; \lVert x\rVert_2^2 + \lVert c\rVert_2^2 \;-\; 2\,x^\top c.
```

### Implementation Notes

* **k-means++** init + **multi-restart** (`n_init`) for robustness.
* Empty clusters are reseeded to the **globally farthest** point (stabilizes purity).
* API: `fit`, `predict`, `fit_predict`; stores `cluster_centers_`, `inertia_`.

### Quick Commands

```bash
# Blobs with k=3 clusters; report inertia and purity (majority-assign accuracy)
uv run python scripts/run_kmeans.py --n 600 --k 3 --d 2 --sep 4.0
# → KMeans: inertia=...  purity=0.9xx
```

**Purity** here maps each predicted cluster to its majority true label and reports overall accuracy.

---

## Gaussian Mixture Models (GMM, EM)

**Paths**

* Code: `src/airoad/unsupervised/gmm.py`
* Script: `scripts/run_gmm.py`
* Test: `tests/test_gmm_nmi.py`

### Intuition

GMM models data as a mixture of $K$ Gaussians. Parameters ${\pi_k,\mu_k,\Sigma_k}_{k=1}^{K}$ are estimated by **EM**:

* **E-step:** responsibilities $r_{nk} \propto \pi_k ,\mathcal{N}(x_n \mid \mu_k, \Sigma_k)$
* **M-step:** update $\pi_k,\mu_k,\Sigma_k$ from the soft counts $r_{nk}$

### EM (sketch)

```math
r_{nk} \;=\; \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}
                  {\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)},\qquad
N_k \;=\; \sum_{n=1}^{N} r_{nk}.
```

```math
\pi_k \leftarrow \frac{N_k}{N},\quad
\mu_k \leftarrow \frac{1}{N_k}\sum_{n} r_{nk} x_n,\quad
\Sigma_k \leftarrow \frac{1}{N_k}\sum_{n} r_{nk}(x_n - \mu_k)(x_n - \mu_k)^\top + \lambda I.
```

### Implementation Notes

* **Initialization:** K-Means (k-means++ + multi-restart) for robust seeds (means/weights/covariances).
* **Covariance:** diagonal/full/spherical options; diagonal or full are typical.
* **Stability:** log-sum-exp for responsibilities; `reg_covar` for PD covariance.
* **Multi-init EM:** run several seeds; keep the **best log-likelihood**; fallback to the best init if EM ever degrades LL.
* Optional **whitening** (standardization) can help on anisotropic data.

### Quick Commands

```bash
uv run python scripts/run_gmm.py
# → GMM NMI=0.9xx   (Normalized Mutual Information vs. ground truth)
```

---

## PCA → K-Means Pipeline

Reduce dimension / denoise with PCA, then cluster the **compressed** representation.

### Why it helps

* PCA removes low-variance directions (often mostly noise).
* Lower dimension → faster distances and fewer poor local minima.

### Minimal Example

```python
from airoad.unsupervised.pca import PCA
from airoad.unsupervised.kmeans import KMeans

# X: (n,d)
Z = PCA(n_components=2).fit_transform(X)    # (n,2)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z)
```

Try both raw and PCA-reduced features; compare **inertia** and external metrics (purity, **NMI**, **ARI**).

---

## Tests & Acceptance Gates

* **PCA** (`tests/test_pca.py`)

  * Top-2 EVR $> 0.5$ on correlated data
  * Recon MSE below a fraction of total variance

* **K-Means** (`tests/test_kmeans.py`)

  * On well-separated blobs, **purity $\ge 0.9$** with k-means++ and multi-restart

* **GMM (EM)** (`tests/test_gmm_nmi.py`)

  * On well-separated blobs, **NMI $\ge 0.9$** (label-permutation invariant)

Run all:

```bash
uv run pytest -q
```

---

## Common Pitfalls & Debugging

* **Feature scale:** PCA is scale-sensitive; standardize first when units differ.
* **K-Means local minima:** increase `n_init`, check `inertia_`, visualize in 2D.
* **Empty clusters (K-Means):** reseed to the globally farthest point; increase `n_init`.
* **GMM init:** poor seeds → mediocre EM basin; strengthen K-Means restarts (larger `n_init`) and allow a few more EM steps.
* **Covariance choice (GMM):** on axis-aligned blobs, **diag** often beats spherical; on richer structure, **full** works if you have enough points.

---

## Next Steps

* **PCA variants:** whitening; randomized/streaming/truncated SVD for large $n,d$.
* **More clustering:** Spectral Clustering, DBSCAN/HDBSCAN.
* **Mixtures:** mixture of t-distributions (heavy-tailed) for outlier robustness.
* **Embeddings → clustering:** sentence embeddings → PCA → K-Means to discover topics.
* **Visualization:** t-SNE/UMAP (for small $n$) to explore cluster geometry in 2D.

---

## File Map (Unsupervised)

* `src/airoad/unsupervised/pca.py` — SVD-based PCA (`fit/transform/inverse_transform`)
* `src/airoad/unsupervised/kmeans.py` — K-Means with k-means++ and multi-restart
* `src/airoad/unsupervised/gmm.py` — GMM (EM) with K-Means init + multi-init EM
* `scripts/run_pca.py` — PCA quick demo (EVR, recon MSE)
* `scripts/run_kmeans.py` — K-Means quick demo (inertia, purity)
* `scripts/run_gmm.py` — GMM demo (NMI)
* `tests/test_pca.py`, `tests/test_kmeans.py`, `tests/test_gmm_nmi.py` — tiny, fast gates

---

**Keep iterating:** small, fast experiments build durable intuition. When you’re ready, try a **PCA→K-Means→GMM** comparison notebook and visualize cluster boundaries vs. component densities.
