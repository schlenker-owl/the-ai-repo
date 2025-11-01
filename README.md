# the-ai-repo (package: `airoad`) — Hands-On AI/ML/LLMs Roadmap

Learn modern **Machine Learning** and **LLMs** by building them from the ground up. This repo is **local-first** (Apple Silicon **MPS**/CPU), **uv-managed**, and organized as fast, reproducible labs that run in minutes. Each module includes small, focused scripts and tiny tests so you understand *how and why* things work before scaling.

> **Import name:** `airoad` (under `src/airoad`)  
> **Distribution name (project):** `airepo` (in `pyproject.toml`)

---

## ✨ Highlights

- **Local-first**: Runs out-of-the-box on macOS (M-series) or CPU; tiny datasets; short wall-clock.
- **Modern packaging**: **uv** for venv + dependency locking (`uv.lock`), fast installs, simple CI.
- **Readable “from-scratch” implementations**: NumPy gradient descent, ridge/lasso (closed-form & coord-descent), logistic regression, and optimizer lab (SGD/Momentum/Adam).
- **Batteries included**: Scripts to train a tiny Transformer on Tiny Shakespeare, a minimal RAG demo with FAISS (optional extra), and quick tests that run in < 1–2 seconds.
- **Open-source friendly**: MIT license, CONTRIBUTING, CODE_OF_CONDUCT, dataset/model card templates.

---

## 🧱 Repository Structure

```

.
├── pyproject.toml          # uv-managed project; dist name = "airepo"; import = "airoad"
├── README.md
├── src/
│   └── airoad/
│       ├── **init**.py
│       ├── datasets/
│       │   └── toy.py                  # synthetic regression/classification + standardize()
│       ├── models/
│       │   ├── linreg_numpy.py         # linear regression (GD)
│       │   ├── logreg_numpy.py         # logistic regression (GD)
│       │   └── ridge_lasso.py          # ridge (closed-form/GD), lasso (coord-descent)
│       ├── optimizers/
│       │   └── optimizers.py           # SGD, Momentum, Adam + stable logistic_loss_grad()
│       └── utils/
│           ├── device.py               # MPS/cuda/cpu picker (used later)
│           └── seed.py                 # reproducible seeding helpers
├── scripts/
│   ├── run_linreg.py                   # tiny linear regression demo
│   ├── run_logreg.py                   # tiny logistic regression demo
│   ├── run_ridge_lasso.py              # ridge/lasso demos
│   ├── optimizer_lab.py                # compare SGD/Momentum/Adam on a toy task
│   ├── fetch_data.py                   # tiny text corpora (Tiny Shakespeare / toy)
│   ├── train_tiny_transformer.py       # tiny character LM with HF Trainer
│   └── rag_build_faiss.py              # minimal FAISS index (optional extra)
└── tests/
├── test_imports.py
├── test_smoke_env.py
├── test_linreg_grad.py
├── test_logreg_perf.py
├── test_ridge_lasso.py
└── test_optimizers.py

````

---

## 🚀 Quickstart (macOS Apple Silicon / CPU, using **uv**)

> Requires Python **3.11+**. If you don’t have `uv`, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

Create and sync the project environment:

```bash
# from repo root
uv venv --python 3.11
source .venv/bin/activate

# install core + dev/lint groups (creates/updates uv.lock)
uv sync --all-groups

# editable install and pre-commit hooks
uv run python -m pip install -e .
uv run pre-commit install

# (optional) Jupyter kernel
uv run python -m ipykernel install --user --name airoad --display-name "Python (airoad)"

# Sanity checks
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY

uv run pytest -q   # tests should pass in ~1s
```

**VSCode tip:** Command Palette → *Python: Select Interpreter* → `${workspaceFolder}/.venv/bin/python`.

---

## 🧪 First Labs (run in seconds)

Linear Regression (NumPy, GD):

```bash
uv run python scripts/run_linreg.py --n 200 --d 1 --noise 0.1 --lr 0.1 --epochs 500
```

Logistic Regression (NumPy, GD):

```bash
uv run python scripts/run_logreg.py --n 300 --margin 0.5 --lr 0.2 --epochs 800
```

Ridge (closed-form vs GD) and Lasso (coordinate descent):

```bash
uv run python scripts/run_ridge_lasso.py ridge --lam 0.5 --epochs 300
uv run python scripts/run_ridge_lasso.py lasso --lam 0.3
```

Optimizer Lab (SGD vs Momentum vs Adam):

```bash
uv run python scripts/optimizer_lab.py --steps 300
```

---

## 🧠 Tiny Language Model & RAG Demo (optional extras)

**Tiny Transformer** (HF Trainer on Tiny Shakespeare / toy fallback):

```bash
uv run python scripts/fetch_data.py tinyshakespeare
uv run python scripts/train_tiny_transformer.py --max-steps 300 --batch-size 2
```

**RAG (FAISS)** — enable extras and build a tiny index:

```bash
# install extras (once) or sync with extras on CI/colleagues' machines
uv sync --all-groups --extra rag

uv run python scripts/rag_build_faiss.py
```

> If `faiss-cpu` wheel isn’t available for your exact platform, the script falls back to scikit-learn nearest neighbors so you’re never blocked.

---

## 📦 Project & Packaging Notes

* **Distribution name:** `airepo` (in `pyproject.toml`).
* **Import name:** `airoad` (your code imports `import airoad`).
* **Lockfile:** Commit `uv.lock` for reproducibility; in CI use `uv sync --locked`.

Minimal import sanity check:

```python
import airoad
print(airoad.about())
```

---

## 🧭 Curriculum (Month-long, Local-first)

**Week 1 — Foundations & Classics**

* Linear vs logistic regression; regularization (ridge/lasso); bias-variance; gradient checks.

**Week 2 — Deep Learning Fundamentals**

* Autodiff intuition; NumPy→PyTorch MLP; optimizers (SGD/Momentum/Adam); schedulers.

**Week 3 — Transformers & Pretraining**

* Attention → Transformer block; tokenization; tiny character LM; throughput tips (batching, seq len).

**Week 4 — LLM Systems in Practice**

* PEFT (LoRA/QLoRA) concept lab, basic eval (accuracy/perplexity), minimal RAG with FAISS; serving preview.

> Every lab aims for ≤ 10–20 minutes on Apple Silicon/CPU with observable learning signals (loss ↓, accuracy ↑).

---

## 🔧 Developer Workflow

Formatting & Lint:

```bash
uv run ruff check .
uv run black --check .
uv run isort --check-only .
```

Fix style:

```bash
uv run ruff check . --fix
uv run black .
uv run isort .
```

Run all tests:

```bash
uv run pytest -q
```

---

## 📈 Observability (optional)

* We’ll add **Weights & Biases** hooks when you enable the `llm` extra:

  ```bash
  uv sync --all-groups --extra llm
  ```

  Then wire a few lines in scripts to log metrics/artifacts when desired.

---

## 🔐 Data, Safety, and Licensing

* **Data governance:** Use `docs/dataset_card_template.md` for any added datasets. Avoid restricted/PII data.
* **Model card:** `docs/model_card_template.md` for checkpoints you produce.
* **Secrets:** Never commit tokens/keys. Use environment variables.

---

## 🤝 Contributing

1. `uv venv && uv sync --all-groups`
2. `uv run python -m pip install -e .` + `uv run pre-commit install`
3. `uv run pytest -q` must pass before PR.
4. Keep labs small & fast; add tests that run in < 2 seconds.

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

---

## 📄 License

**MIT** — see `LICENSE`.

---

## ❓ FAQ

**Q:** *uv vs conda?*
**A:** This repo standardizes on **uv** for speed and lockfiles. You can keep conda for other work; just ensure VSCode uses `.venv/bin/python` here.

**Q:** *Do I need a GPU?*
**A:** No. Apple Silicon **MPS** accelerates PyTorch automatically. Some ops may fall back to CPU (fine for these labs).

**Q:** *FAISS failed to install—now what?*
**A:** The RAG script falls back to scikit-learn neighbors. You can keep learning without FAISS; enable the `rag` extra when wheels are available.

---

Happy building! If you want the next modules (Decision Tree vs Linear SVM from scratch, then a PyTorch MLP bridge), open an issue or ask in chat and we’ll scaffold them with tests and runnable scripts.

```
::contentReference[oaicite:0]{index=0}
```
