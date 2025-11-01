````markdown
# Documentation — Foundations Roadmap (v0.1)

Welcome! This learning guide walks you through everything already implemented in **the-ai-repo** (import package: `airoad`). It’s designed for fast, local experiments (Apple Silicon **MPS**/CPU) that build intuition first, then deepen.

> If you prefer a “do-first” style: jump to **[Quickstart Labs](#quickstart-labs)** and run them. Each lab takes minutes.

---

## Table of Contents

1. [Environment & Project Basics](#environment--project-basics)
2. [Datasets & Standardization](#datasets--standardization)
3. [Linear Regression (NumPy, GD)](#linear-regression-numpy-gd)
4. [Logistic Regression (NumPy, GD)](#logistic-regression-numpy-gd)
5. [Ridge (Closed-Form & GD) and Lasso (Coordinate Descent)](#ridge-closedform--gd-and-lasso-coordinate-descent)
6. [Optimizer Lab: SGD vs Momentum vs Adam](#optimizer-lab-sgd-vs-momentum-vs-adam)
7. [Decision Tree (Gini) with Two-Level Lookahead](#decision-tree-gini-with-twolevel-lookahead)
8. [Linear SVM (Hinge + L2)](#linear-svm-hinge--l2)
9. [PyTorch MLP (Autodiff Bridge)](#pytorch-mlp-autodiff-bridge)
10. [Tiny Transformer & Mini RAG Demo (Optional)](#tiny-transformer--mini-rag-demo-optional)
11. [Testing & Acceptance Gates](#testing--acceptance-gates)
12. [Quickstart Labs](#quickstart-labs)
13. [What to Try Next](#what-to-try-next)

---

## Environment & Project Basics

- **Dependency manager**: [`uv`](https://docs.astral.sh/uv) (fast venv, lockfile).
- **Import package**: `airoad` (under `src/airoad`).
- **Distribution name**: `airepo` (in `pyproject.toml`).
- **Device**: Apple Silicon **MPS** or CPU (works out-of-the-box).

Setup:
```bash
# from repo root
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pre-commit install
uv run pytest -q
````

VSCode → Command Palette → **Python: Select Interpreter** → `.venv/bin/python`.

---

## Datasets & Standardization

**Where**: `src/airoad/datasets/toy.py`

* `make_linear_regression(n,d,noise)` → synthetic y = Xw + b + noise
* `make_classification_2d(n, margin)` → binary labels with adjustable separability
* `standardize(X)` → z-score per feature (critical for stable gradient descent)

**Why**: Standardization reduces condition numbers → smoother, faster convergence.

---

## Linear Regression (NumPy, GD)

**Where**: `src/airoad/models/linreg_numpy.py`
**Script**: `scripts/run_linreg.py`

**Objective**
[
\min_{w,b}; \frac{1}{n}|Xw + b - y|^2
]

**Gradient (weights-with-bias formulation)**
With bias appended to X as a last column of ones:
[
\nabla_W = \frac{2}{n} X_{\text{ext}}^\top (X_{\text{ext}}W - y)
]

**Mental Model**:

* You’re minimizing squared error by following the slope of the loss surface.
* Standardization lets a single learning rate work across features.

**Try**:

```bash
uv run python scripts/run_linreg.py --n 200 --d 1 --noise 0.1 --lr 0.1 --epochs 500
```

**Notice**: MSE falls; learned `w,b` close to true.

---

## Logistic Regression (NumPy, GD)

**Where**: `src/airoad/models/logreg_numpy.py`
**Script**: `scripts/run_logreg.py`

**Objective**
[
\min_{w,b}; -\frac{1}{n}\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)] + \lambda|w|_2^2,\quad p_i=\sigma(X_i w + b)
]

**Gradient**
[
\nabla_w = \frac{1}{n} X^\top (p - y) + 2\lambda w,\quad \nabla_b = \text{mean}(p-y)
]

**Why**: Probabilistic view; outputs calibrated probabilities (unlike SVM).

**Try**:

```bash
uv run python scripts/run_logreg.py --n 300 --margin 0.5 --lr 0.2 --epochs 800
```

**Notice**: Accuracy increases; standardization helps stability.

---

## Ridge (Closed-Form & GD) and Lasso (Coordinate Descent)

**Where**: `src/airoad/models/ridge_lasso.py`
**Script**: `scripts/run_ridge_lasso.py`

**Ridge Objective**
[
\min_{w,b}; \frac{1}{n}|Xw+b-y|^2 + \lambda|w|_2^2
]

* **Closed-form** (normal equations with bias unpenalized).
* **GD** (same gradient form + L2 term).

**Lasso Objective (CD)**
[
\min_{w,b}; \frac{1}{2n}|y-(Xw+b)|^2 + \lambda |w|_1
]

* Coordinate descent with **soft-thresholding** → sparsity grows with λ.

**Try**:

```bash
# Ridge: closed-form vs GD match
uv run python scripts/run_ridge_lasso.py ridge --lam 0.5 --epochs 300

# Lasso: see coefficients zero-out as λ grows
uv run python scripts/run_ridge_lasso.py lasso --lam 0.3
```

**Notice**: Ridge shrinks weights; Lasso zeros them → feature selection.

---

## Optimizer Lab: SGD vs Momentum vs Adam

**Where**: `src/airoad/optimizers/optimizers.py`
**Script**: `scripts/optimizer_lab.py`

* **SGD**: simplest; can zig-zag in ravines.
* **Momentum**: averages gradients → smoother updates.
* **Adam**: adaptive step sizes per parameter; often fastest to “good enough”.

**Try**:

```bash
uv run python scripts/optimizer_lab.py --steps 300
```

**Notice**: Relative final losses (Adam usually ≤ Momentum ≤ SGD on our toy).

---

## Decision Tree (Gini) with Two-Level Lookahead

**Where**: `src/airoad/models/tree_gini.py`

* Greedy CART-style splits maximize **Gini gain**.
* **XOR** is not separable by a single axis-aligned split at the root.
* We **compare** greedy gain with a **two-level lookahead** plan and choose the better—this lets the tree crack XOR at depth 2–3.

**Takeaway**: Trees fit non-linear boundaries but can overfit; depth/leaf size control variance.

---

## Linear SVM (Hinge + L2)

**Where**: `src/airoad/models/linear_svm.py`

**Objective (primal)**
[
\min_w \frac{\lambda}{2}|w|_2^2 + \frac{1}{n}\sum_i \max(0, 1 - y_i f(x_i)),\quad y_i\in{-1,1}
]

* Margin-based; unlike logistic, not probabilistic.
* We use a deterministic, full-batch subgradient (Pegasos-style spirit), bias **unpenalized**.

**When SVM > Logistic**: linearly separable data with clear margins; robust to outliers on the score side.

---

## PyTorch MLP (Autodiff Bridge)

**Where**:

* Model: `src/airoad/dl/mlp_torch.py`

* Script: `scripts/train_mlp_torch.py`

* **Why**: Move from manual NumPy gradients to **autograd**.

* **Task**: Learn XOR (non-linear) with a small MLP; loss `BCEWithLogitsLoss`.

**Try**:

```bash
uv run python scripts/train_mlp_torch.py --steps 300
```

**Notice**: High accuracy with tiny network; ReLU layers create non-linear decision boundaries.

---

## Tiny Transformer & Mini RAG Demo (Optional)

**Where**:

* Transformer: `scripts/train_tiny_transformer.py` (character LM, Tiny Shakespeare or toy fallback)
* RAG: `scripts/rag_build_faiss.py` (builds minimal FAISS index; falls back to sklearn if FAISS wheel missing)

**Enable extras once**:

```bash
uv sync --all-groups --extra rag   # for FAISS + sentence-transformers
```

**Try**:

```bash
uv run python scripts/fetch_data.py tinyshakespeare
uv run python scripts/train_tiny_transformer.py --max-steps 300 --batch-size 2
uv run python scripts/rag_build_faiss.py
```

---

## Testing & Acceptance Gates

**Where**: `tests/`

* `test_linreg_grad.py` → analytic ≈ numerical gradient check (tight tolerance).
* `test_logreg_perf.py` → logistic reaches **≥ 0.9** accuracy on easy data.
* `test_ridge_lasso.py` → ridge (closed-form ≈ GD), lasso zeros increase with λ.
* `test_optimizers.py` → Adam not worse than SGD on toy.
* `test_tree_gini.py` → tree cracks XOR with lookahead.
* `test_linear_svm.py` → SVM ≥ 0.9 on separable set.
* `test_mlp_torch.py` → MLP ≥ 0.9 on XOR within ~120 steps (CPU).

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

# 5) Decision tree & SVM (run tests or plug into your own data)
uv run pytest -q

# 6) PyTorch MLP (autodiff)
uv run python scripts/train_mlp_torch.py --steps 300

# 7) Optional: Tiny Transformer & RAG index
uv run python scripts/fetch_data.py tinyshakespeare
uv run python scripts/train_tiny_transformer.py --max-steps 300 --batch-size 2
uv sync --all-groups --extra rag
uv run python scripts/rag_build_faiss.py
```

---

## What to Try Next

**Level up the Foundations**

* **Regularization paths**: sweep λ for ridge/lasso; plot MSE vs sparsity.
* **Decision boundary gallery**: compare Logistic vs SVM vs Tree vs MLP on the same 2D datasets.
* **Optimizer dynamics**: track loss curves per step; visualize gradient norms and effective step sizes.

**Bridge to Deep Learning**

* Implement a **NumPy MLP** (manual backprop) mirroring the PyTorch MLP.
* Add a **learning-rate scheduler** & **weight decay** to MLP training script.

**Toward LLM Systems**

* Add a **toy tokenizer** (BPE-lite) to see how subwords form.
* Extend the Transformer script with **packing** and **gradient clipping**; log tokens/s.
* Expand RAG with a small **retrieval + prompt** loop and a minimal evaluation rubric.

---

### File Map (for reference)

* `src/airoad/datasets/toy.py` — synthetic data + standardization
* `src/airoad/models/linreg_numpy.py` — linear regression (GD)
* `src/airoad/models/logreg_numpy.py` — logistic regression (GD)
* `src/airoad/models/ridge_lasso.py` — ridge (closed-form & GD), lasso (CD)
* `src/airoad/models/tree_gini.py` — decision tree (Gini + lookahead)
* `src/airoad/models/linear_svm.py` — linear SVM (hinge + L2)
* `src/airoad/optimizers/optimizers.py` — SGD, Momentum, Adam, stable logistic loss
* `src/airoad/dl/mlp_torch.py` — simple MLP (PyTorch)
* `scripts/*.py` — runnable labs
* `tests/*.py` — tiny, fast gates

---

**You’re ready.** Pick a lab, run it, observe, and tweak. The repo is built for short, meaningful learning cycles—stack them, and you’ll build deep intuition quickly.

```
::contentReference[oaicite:0]{index=0}
```
