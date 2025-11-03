# LLM Systemization — LoRA SFT · RAG Eval · OpenAI-Compatible Server (v0.2)

This document explains the **LLM systemization** pieces in **the-ai-repo** (`airoad`):
- **LoRA SFT** on a tiny model (CPU/MPS-friendly)  
- **Inference helpers** (base vs LoRA, adapter merge) + **before/after** comparison  
- **RAG evaluation** (Exact Match, cosine similarity, ROUGE-L) with TF-IDF fallback  
- **OpenAI-compatible server** for quick local smoke tests

> **Math on GitHub:** inline math uses `$…$`; display equations use fenced code blocks with `math`.

---

## Contents
1. [Environment](#environment)  
2. [LoRA SFT (Tiny, CPU/MPS-Friendly)](#lora-sft-tiny-cpumps-friendly)  
3. [Inference, Adapters, and Merge](#inference-adapters-and-merge)  
4. [RAG Evaluation (EM, Cosine, ROUGE-L)](#rag-evaluation-em-cosine-rouge-l)  
5. [OpenAI-Compatible Server](#openai-compatible-server)  
6. [Quickstart Commands](#quickstart-commands)  
7. [Tests & Acceptance Gates](#tests--acceptance-gates)  
8. [File Map](#file-map)  
9. [Next Steps](#next-steps)

---

## Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pytest -q
````

If you plan to run LoRA or the server:

```bash
uv pip install transformers peft  # optional: trl sentence-transformers fastapi uvicorn scikit-learn
```

---

## LoRA SFT (Tiny, CPU/MPS-Friendly)

**Paths**

* Trainer (version-robust): `src/airoad/sft/lora_sft.py`
* Script: `scripts/sft_lora.py`

We fine-tune a tiny Causal LM (e.g., `sshleifer/tiny-gpt2`) on a small, in-repo instruction set using **LoRA**. The core LM objective (teacher forcing) is:

```math
\mathcal{L}_{\text{CausalLM}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
```

LoRA injects a **low-rank** adapter into linear projections:

```math
W^\star
= W + \Delta W,\quad
\Delta W = A B,\quad
A \in \mathbb{R}^{d \times r},\; B \in \mathbb{R}^{r \times k},\; r \ll \min(d,k).
```

This reduces trainable parameters while preserving base weights.

---

## Inference, Adapters, and Merge

**Paths**

* Inference helpers: `src/airoad/sft/infer.py`
* Before/After qualitative compare: `scripts/sft_compare_generate.py`
* Toy SFT evaluation (quant): `scripts/sft_eval_toy.py`

With `infer.py` you can:

* **Load base** model & tokenizer
* **Load base + LoRA** adapters (PEFT)
* **Merge adapters** into base weights (for deployment)
* **Generate** continuations with simple settings

---

## RAG Evaluation (EM, Cosine, ROUGE-L)

**Paths**

* RAG eval utilities: `src/airoad/rag/eval.py`
* Demo: `scripts/rag_eval_demo.py`

We provide small, dependency-light metrics:

**Exact Match (EM)** (case-insensitive):

```math
\mathrm{EM}(\hat{y}, y) \;=\; \mathbf{1}\{\mathrm{lower}(\hat{y}) = \mathrm{lower}(y)\}.
```

**Cosine Similarity** (ST embeddings or TF-IDF fallback):

```math
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\lVert \mathbf{a} \rVert \,\lVert \mathbf{b} \rVert }.
```

**ROUGE-L F-score** (token LCS):

```math
\begin{aligned}
\mathrm{LCS}(\hat{y}, y) &= \text{longest common subsequence},\\
\mathrm{prec} &= \frac{\mathrm{LCS}}{|\hat{y}|},\quad
\mathrm{rec} = \frac{\mathrm{LCS}}{|y|},\\
\mathrm{ROUGE\text{-}L\_F} &= \frac{2\cdot \mathrm{prec}\cdot \mathrm{rec}}{\mathrm{prec} + \mathrm{rec} + \epsilon}.
\end{aligned}
```

Retrieval uses normalized embeddings when available; otherwise TF-IDF with cosine on sparse vectors.

---

## OpenAI-Compatible Server

**Path**

* `scripts/serve_openai_compat.py`

A tiny local server exposing `POST /v1/chat/completions`. It backs responses with:

* Your **GPTTiny** char-model (if importable), or
* An **echo fallback** when the tiny model isn’t available.

This is for **smoke testing** clients or basic integration demos.

---

## Quickstart Commands

**LoRA SFT (tiny)**

```bash
uv run python scripts/sft_lora.py --max-steps 50 --base-model sshleifer/tiny-gpt2
```

**Before/After generation**

```bash
uv run python scripts/sft_compare_generate.py \
  --base-model sshleifer/tiny-gpt2 --lora-dir outputs/sft-lora
# Merge adapters then compare
uv run python scripts/sft_compare_generate.py \
  --base-model sshleifer/tiny-gpt2 --lora-dir outputs/sft-lora --use-merged True
```

**Toy SFT evaluation (EM + ROUGE-L)**

```bash
uv run python scripts/sft_eval_toy.py \
  --base-model sshleifer/tiny-gpt2 --lora-dir outputs/sft-lora
```

**RAG demo (retrieval + metrics)**

```bash
uv run python scripts/rag_eval_demo.py
```

**OpenAI-compatible server**

```bash
uv run python scripts/serve_openai_compat.py --host 0.0.0.0 --port 8000
# curl example:
# curl http://localhost:8000/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{"model":"gpttiny","messages":[{"role":"user","content":"Say hello!"}]}'
```

---

## Tests & Acceptance Gates

* **SFT prep** (`tests/test_sft_prep.py`) — formatting & dataset generation; **no downloads**.
* **SFT eval metrics** (`tests/test_sft_eval_metrics.py`) — EM and ROUGE-L shape/range.
* **RAG metrics** (`tests/test_rag_eval_metrics.py`) — TF-IDF fallback ranks the right doc; `evaluate_qa` fields intact.

> LoRA training itself is not unit-tested (needs network); run the script manually, then evaluate with the toy metrics.

---

## File Map

* **SFT (training + infer)**

  * `src/airoad/sft/lora_sft.py` — LoRA SFT (TRL or plain HF `Trainer`, version-robust)
  * `src/airoad/sft/infer.py` — load base, load+LoRA, merge, generate
  * `src/airoad/sft/eval_sft.py` — EM & ROUGE-L metrics
  * `scripts/sft_lora.py` — train LoRA on a tiny base
  * `scripts/sft_compare_generate.py` — before/after qualitative comparison
  * `scripts/sft_eval_toy.py` — quantitative toy eval (EM + ROUGE-L)

* **RAG**

  * `src/airoad/rag/eval.py` — exact match, cosine, retrieval (ST/TF-IDF), `evaluate_qa`
  * `scripts/rag_eval_demo.py` — tiny retrieval + scoring demo

* **Server**

  * `scripts/serve_openai_compat.py` — minimal `/v1/chat/completions` endpoint

* **Tests**

  * `tests/test_sft_prep.py`, `tests/test_sft_eval_metrics.py`, `tests/test_rag_eval_metrics.py`

---

## Next Steps

* **Adapter merging demo** already supported — add a **“merged-only inference”** script if preferred.
* Add a tiny **RAG generate** loop (retriever + generator) and score with `evaluate_qa`.
* Introduce **prompt templates** (Alpaca / ChatML) and **stop sequences** for cleaner generations.
* Add **quantization** notes (4-bit/8-bit) for larger bases when you move beyond tiny models.
* (Later) Upgrade to transformers v5 and switch `Trainer(..., tokenizer=tok)` → `processing_class=tok`.

**Goal:** keep everything **local-first**, **minutes-to-run**, and **educational** — the same rhythm your repo nails across ML/DL/LLMs.

