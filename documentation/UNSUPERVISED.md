# Unsupervised Learning — PCA & K-Means (v0.1)

This guide explains the **unsupervised** modules currently implemented in **the-ai-repo** (`airoad`). Each section gives you: intuition → math (GitHub-friendly) → implementation notes → quick commands → tiny acceptance checks.

> **Math rendering on GitHub:** inline formulas go in `$…$` and display equations go in `$$…$$`.

---

## Contents

1. [Environment](#environment)
2. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [K-Means Clustering](#k-means-clustering)
4. [PCA → K-Means Pipeline](#pca--k-means-pipeline)
5. [Tests & Acceptance Gates](#tests--acceptance-gates)
6. [Common Pitfalls & Debugging](#common-pitfalls--debugging)
7. [Next Steps](#next-steps)
8. [File Map (Unsupervised)](#file-map-unsupervised)

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
X_c \;=\; X - \mathbf{1}\,\mu^\top
```

2. **SVD** (economy):

```math
X_c \;=\; U\,S\,V^\top, \qquad
\text{PC directions} = V, \quad
\text{PC variances} = \frac{S^2}{\,n-1\,}
```

3. **Keep top-$k$** rows of $V^\top$ as $W_k \in \mathbb{R}^{d \times k}$ and **project**:

```math
Z \;=\; X_c\,W_k \;\in\; \mathbb{R}^{n \times k}
```

4. **Reconstruct** (optional):

```math
\hat{X} \;=\; Z\,W_k^\top + \mu
```

5. **Explained variance ratio** (EVR) for top-$k$:

```math
\text{EVR}_k \;=\; \frac{\sum_{i=1}^{k} S_i^2}{\sum_{i=1}^{\min(n,d)} S_i^2}
```

### Implementation Notes

* Uses **SVD** (stable) rather than covariance eigendecomposition.
* Stores: `mean_`, `components_ (k×d)`, `explained_variance_`, `explained_variance_ratio_`.
* API: `fit`, `transform`, `inverse_transform`, `fit_transform`.

### Quick Commands

```bash
# Run PCA on correlated synthetic data, report variance ratio & recon error
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

Given centers $C={c_1,\dots,c_k}$ and labels $\ell_i\in{1,\dots,k}$,

```math
\min_{C,\;\ell}\;\sum_{i=1}^{n} \big\lVert x_i - c_{\ell_i} \big\rVert_2^2
```

The objective value at the solution is often called **inertia**.

### Algorithm (with k-means++)

1. **Init** with **k-means++** (diverse seeds; fewer bad local minima).
2. **Lloyd iterations** until convergence:

   * Assign each point to the nearest center.
   * Update each center to the **mean** of its assigned points.
   * Handle empty clusters by reseeding (globally farthest point).
3. Run **multiple restarts** (`n_init`) and keep the best inertia.

Useful identity for fast distances:

```math
\lVert x - c \rVert_2^2 \;=\; \lVert x\rVert_2^2 + \lVert c\rVert_2^2 - 2\,x^\top c
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

**Purity** is computed against synthetic ground-truth labels by mapping each predicted cluster to its majority true label.

---

## PCA → K-Means Pipeline

Reduce dimension / denoise with PCA, then cluster the **compressed** representation.

### Why it helps

* PCA removes low-variance directions (often mostly noise).
* Lower dimension → faster distances and fewer poor local minima.

### Minimal Example

```python
import numpy as np
from airoad.unsupervised.pca import PCA
from airoad.unsupervised.kmeans import KMeans

# X: (n,d) data you loaded
pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)                # (n,2)
km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(Z)
labels = km.predict(Z)
```

Try both raw features and PCA-reduced features; compare **inertia** and any external metric (e.g., purity, ARI, NMI).

---

## Tests & Acceptance Gates

* **PCA** (`tests/test_pca.py`)

  * Top-2 EVR $> 0.5$ on correlated data.
  * Reconstruction MSE below a fraction of total variance.

* **K-Means** (`tests/test_kmeans.py`)

  * On clean, well-separated blobs, **purity $\ge 0.9$** with k-means++ and multi-restart.

Run all:

```bash
uv run pytest -q
```

---

## Common Pitfalls & Debugging

* **Feature scale** varies wildly → center/standardize before PCA or K-Means.
* **Local minima** in K-Means → increase `n_init`, check `inertia_`, visualize in 2D.
* **Empty clusters** → our reseed strategy helps; still, try higher `n_init`.
* **PCA component sign flips** → mathematically equivalent; don’t treat sign as semantic.

---

## Next Steps

* **PCA options:** whitening; randomized/streaming/truncated SVD for large $n,d$.
* **More clustering:** Gaussian Mixture Models (EM), DBSCAN/HDBSCAN, Spectral Clustering.
* **Embeddings pipeline:** sentence embeddings → PCA → K-Means to discover topics.
* **Visualization:** t-SNE/UMAP (for small $n$) to see clusters and PCA axes interact.

---

## File Map (Unsupervised)

* `src/airoad/unsupervised/pca.py` — SVD-based PCA (`fit/transform/inverse_transform`)
* `src/airoad/unsupervised/kmeans.py` — K-Means with k-means++ and multi-restart
* `scripts/run_pca.py` — PCA quick demo (variance ratio, recon MSE)
* `scripts/run_kmeans.py` — K-Means quick demo (inertia, purity)
* `tests/test_pca.py`, `tests/test_kmeans.py` — tiny, fast gates

---

**Keep iterating:** small, fast experiments build durable intuition. When you’re ready, we’ll add **GMM (EM)** and a **PCA→K-Means notebook** with plots to compare raw vs. reduced clustering.


