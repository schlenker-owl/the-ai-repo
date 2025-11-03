# Documentation — Foundations Roadmap (v0.1)

Welcome! This learning guide covers the **foundational ML & DL** pieces in **the-ai-repo** (import package: `airoad`). It’s built for fast, local experiments on **Apple Silicon (MPS)** or CPU to help you *learn by building*.

> Prefer a “do-first” style? Jump to **[Quickstart Labs](#quickstart-labs)** and run them. Each lab finishes in minutes.

---

## Table of Contents

1. [Environment & Project Basics](#environment--project-basics)  
2. [Datasets & Standardization](#datasets--standardization)  
3. [Linear Regression (NumPy, GD)](#linear-regression-numpy-gd)  
4. [Logistic Regression (NumPy, GD)](#logistic-regression-numpy-gd)  
5. [Ridge (Closed-Form & GD) and Lasso (CD)](#ridge-closed-form--gd-and-lasso-coordinate-descent)  
6. [Optimizer Lab: SGD vs Momentum vs Adam](#optimizer-lab-sgd-vs-momentum-vs-adam)  
7. [Decision Tree (Gini) + Two-Level Lookahead](#decision-tree-gini--twolevel-lookahead)  
8. [Linear SVM (Hinge + L2)](#linear-svm-hinge--l2)  
9. [PyTorch MLP (Autodiff Bridge)](#pytorch-mlp-autodiff-bridge)  
10. [Tiny Transformer & Mini RAG (Optional)](#tiny-transformer--mini-rag-optional)  
11. [Testing & Acceptance Gates](#testing--acceptance-gates)  
12. [Quickstart Labs](#quickstart-labs)  
13. [What to Try Next](#what-to-try-next)

---

## Environment & Project Basics

- **Dependency manager:** [`uv`](https://docs.astral.sh/uv) (fast venv + lockfile)  
- **Import package:** `airoad` (`src/airoad/`)  
- **Distribution name:** `airepo` (see `pyproject.toml`)  
- **Device:** Apple Silicon **M1/M2/M3** (MPS) or CPU

**Setup**
```bash
# from repo root
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pre-commit install
uv run pytest -q
````

> **VS Code:** Command Palette → `Python: Select Interpreter` → `.venv/bin/python`

---

## Datasets & Standardization

**Where:** `src/airoad/datasets/toy.py`

* `make_linear_regression(n, d, noise)` → synthetic regression with
  ( y = X,w + b + \text{noise} )
* `make_classification_2d(n, margin)` → 2-D binary classification with tunable separability
* `standardize(X)` → z-score each feature (mean 0, std 1)

**Why:** z-scoring shrinks pathological condition numbers so gradient steps behave well and converge faster.

---

## Linear Regression (NumPy, GD)

**Where:** `src/airoad/models/linreg_numpy.py`
**Script:** `scripts/run_linreg.py`

**Objective**

```math
\min_{w,b}\; \frac{1}{n}\,\lVert Xw + b - y \rVert_2^2
```

**Gradient (with bias folded into (W) via (X_{\text{ext}})):**

```math
\nabla_W \;=\; \frac{2}{n}\, X_{\text{ext}}^\top\!\big(X_{\text{ext}} W - y\big)
```

**Mental model**

* You’re descending the MSE “bowl”; standardization helps step sizes behave.
* The bias term is handled by augmenting (X) with a column of ones.

**Try**

```bash
uv run python scripts/run_linreg.py --n 200 --d 1 --noise 0.1 --lr 0.1 --epochs 500
```

**Notice:** MSE decreases; learned ( (w, b) ) approaches ground truth.

---

## Logistic Regression (NumPy, GD)

**Where:** `src/airoad/models/logreg_numpy.py`
**Script:** `scripts/run_logreg.py`

**Objective**

```math
\min_{w,b}\; -\frac{1}{n}\sum_{i=1}^{n}\!\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big] \;+\; \lambda \lVert w\rVert_2^2,
\quad p_i = \sigma(x_i^\top w + b)
```

**Gradient**

```math
\nabla_w = \frac{1}{n} X^\top (p - y) + 2\lambda w,
\qquad
\nabla_b = \frac{1}{n}\mathbf{1}^\text{T}(p - y)
```

**Why:** Logistic is a probabilistic classifier; its loss is convex and gives calibrated probabilities (unlike SVM’s hinge).

**Try**

```bash
uv run python scripts/run_logreg.py --n 300 --margin 0.5 --lr 0.2 --epochs 800
```

**Notice:** Accuracy climbs as the decision boundary aligns with the data; z-scaling improves stability.

---

## Ridge (Closed-Form & GD) and Lasso (Coordinate Descent)

**Where:** `src/airoad/models/ridge_lasso.py`
**Script:** `scripts/run_ridge_lasso.py`

**Ridge objective** (bias unpenalized)

```math
\min_{w,b}\; \frac{1}{n}\,\lVert Xw + b - y \rVert_2^2 \;+\; \lambda \lVert w\rVert_2^2
```

* **Closed form:** Solve ( (X^\top X + \lambda I)w = X^\top (y - b) ) via normal equations.
* **GD:** Same gradient as least squares plus (2\lambda w).

**Lasso objective** (coordinate descent)

```math
\min_{w,b}\; \frac{1}{2n}\,\lVert y - (Xw + b) \rVert_2^2 \;+\; \lambda \lVert w\rVert_1
```

* **CD:** Soft-threshold each coordinate to promote exact zeros → sparsity.

**Try**

```bash
# Ridge: closed-form vs GD should closely match
uv run python scripts/run_ridge_lasso.py ridge --lam 0.5 --epochs 300

# Lasso: inspect how many coefficients hit zero as λ grows
uv run python scripts/run_ridge_lasso.py lasso --lam 0.3
```

**Notice:** Ridge shrinks weights; Lasso zeros them for feature selection.

---

## Optimizer Lab — SGD vs Momentum vs Adam

**Where:** `src/airoad/optimizers/optimizers.py`
**Script:** `scripts/optimizer_lab.py`

* **SGD:** simple but can zig-zag.
* **Momentum:** exponentially weighted average of past gradients → smoother steps.
* **Adam:** per-parameter adaptive learning rates; strong default for many DL tasks.

**Try**

```bash
uv run python scripts/optimizer_lab.py --steps 300
```

**Notice:** Final loss typically follows: `Adam ≤ Momentum ≤ SGD` on the toy problem.

---

## Decision Tree (Gini) + Two-Level Lookahead

**Where:** `src/airoad/models/tree_gini.py`

* Greedy CART-style splits maximize Gini gain.
* Purely axis-aligned splits can’t separate XOR *in one step*.
* We add a **two-level lookahead** to evaluate child splits before committing → can solve XOR in shallow trees.

**Takeaway:** Trees model non-linear boundaries but can overfit; control depth, min samples per leaf.

---

## Linear SVM (Hinge + L2)

**Where:** `src/airoad/models/linear_svm.py`

**Primal objective**

```math
\min_{w}\; \frac{\lambda}{2}\,\lVert w\rVert_2^2 \;+\; \frac{1}{n}\sum_{i=1}^{n} \max(0,\;1 - y_i\,x_i^\top w), \quad y_i \in \{-1,1\}.
```

* Margin-based classifier (non-probabilistic).
* Implemented with a deterministic (batch) subgradient update; bias unpenalized.

**When SVM > Logistic:** Clear linear separability or heavy label imbalance/outliers.

---

## PyTorch MLP (Autodiff Bridge)

**Where:** `src/airoad/dl/mlp_torch.py`
**Script:** `scripts/train_mlp_torch.py`

* Two-layer MLP + `BCEWithLogitsLoss` on XOR.
* Shows how autograd replaces manual gradient derivation from the NumPy labs.

**Try**

```bash
uv run python scripts/train_mlp_torch.py --steps 300
```

**Notice:** Rapid drop in loss; non-linear decision boundaries emerge with ReLU layers.

---

## Tiny Transformer & Mini RAG (Optional)

**Where**

* Transformer LM: `scripts/train_gpt_tiny.py` *(or your earlier `scripts/train_tiny_transformer.py`)*
* RAG index demo: `scripts/rag_build_faiss.py` (FAISS)

  > For evaluation utilities (exact-match & cosine) see `docs/LLM_SYSTEMIZATION.md` and `scripts/rag_eval_demo.py`.

**Enable extras (for SentenceTransformers/FAISS)**

```bash
uv sync --all-groups --extra rag
```

**Try**

```bash
uv run python scripts/fetch_data.py tinyshakespeare
uv run python scripts/train_gpt_tiny.py --max-steps 300 --batch-size 2
uv run python scripts/rag_build_faiss.py
```

---

## Testing & Acceptance Gates

**Where:** `tests/`

* `test_linreg_grad.py` — analytic ≈ numerical gradient (tight tol)
* `test_logreg_perf.py` — logistic reaches **≥ 0.90** accuracy on easy data
* `test_ridge_lasso.py` — ridge (closed-form ≈ GD), lasso zeros ↑ with ( \lambda )
* `test_optimizers.py` — Adam ≤ Momentum ≤ SGD on toy
* `test_tree_gini.py` — tree w/ lookahead cracks XOR in few levels
* `test_linear_svm.py` — linear SVM ≥ 0.90 on separable set
* `test_mlp_torch.py` — MLP ≥ 0.90 on XOR within ~120 steps (CPU)

Run:

```bash
uv run pytest -q
```

---

## Quickstart Labs

```bash
# 1) Linear regression (NumPy, GD)
uv run python scripts/run_linreg.py --epochs 500

# 2) Logistic regression (NumPy, GD)
uv run python scripts/run_logreg.py --epochs 800

# 3) Ridge & Lasso
uv run python scripts/run_ridge_lasso.py ridge --lam 0.5 --epochs 300
uv run python scripts/run_ridge_lasso.py lasso --lam 0.3

# 4) Optimizer lab
uv run python scripts/optimizer_lab.py --steps 300

# 5) Trees & SVM (or run the tests below)
uv run pytest -q  # includes tree/SVM/MLP acceptance checks

# 6) PyTorch MLP (autodiff)
uv run python scripts/train_mlp_torch.py --steps 300

# 7) Optional: Tiny Transformer & RAG
uv run python scripts/fetch_data.py tinyshakespeare
uv run python scripts/train_gpt_tiny.py --max-steps 300 --batch-size 2
uv run python scripts/rag_build_faiss.py
```

---

## What to Try Next

**Regularization & geometry**

* Sweep ( \lambda ) for ridge/lasso, plot **MSE vs sparsity**.
* Build a **decision-boundary gallery** (LogReg vs SVM vs Tree vs MLP) on common 2D datasets.
* Instrument the **Optimizer Lab** to log gradient norms and curvature proxies.

**Bridge to DL**

* Add a **NumPy MLP** (manual backprop) mirroring the PyTorch MLP.
* Try **learning-rate schedules** (warmup, cosine) and **weight decay**.

**Unsupervised & LLMs**

* See `docs/UNSUPERVISED.md` (PCA & K-Means) and `docs/GENERATIVE.md` (AE/VAE/DDPM).
* See `docs/LLM_SYSTEMIZATION.md` for **LoRA SFT**, **RAG eval**, and the **OpenAI-compatible server**.

---

## File Map (for reference)

* `src/airoad/datasets/toy.py` — synthetic data + standardization
* `src/airoad/models/linreg_numpy.py` — linear regression (GD)
* `src/airoad/models/logreg_numpy.py` — logistic regression (GD)
* `src/airoad/models/ridge_lasso.py` — ridge (closed-form & GD), lasso (CD)
* `src/airoad/models/tree_gini.py` — decision tree (Gini + lookahead)
* `src/airoad/models/linear_svm.py` — linear SVM (hinge + L2)
* `src/airoad/optimizers/optimizers.py` — SGD, Momentum, Adam (NumPy)
* `src/airoad/dl/mlp_torch.py` — two-layer MLP (PyTorch)
* `scripts/*.py` — runnable labs (see sections above)
* `tests/*.py` — tiny, fast acceptance tests

