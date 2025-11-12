# APPLIED AI LORA

> **Production-ready LoRA/DoRA small-model fine-tuning**: fast probes, steady regular runs, and robust KL-regularized SFT — all with ChatML data, nice logging, and slim, modular code.

---

## Table of Contents

1. [Overview](#overview)
2. [What’s in this repo (LoRA lane)](#whats-in-this-repo-lora-lane)
3. [Prereqs & Environment](#prereqs--environment)
4. [Data formats & prep](#data-formats--prep)
5. [Training — the slim CLI](#training--the-slim-cli)
6. [Training flavors (configs)](#training-flavors-configs)
7. [Merging adapters & inference](#merging-adapters--inference)
8. [Evaluation & style scoring](#evaluation--style-scoring)
9. [Early stopping, time caps, and stability](#early-stopping-time-caps-and-stability)
10. [LoRA vs DoRA, and target placement](#lora-vs-dora-and-target-placement)
11. [KL-regularized SFT](#klregularized-sft)
12. [Performance tips](#performance-tips)
13. [Troubleshooting](#troubleshooting)
14. [Suggested “next steps” (DPO/ORPO/KTO)](#suggested-next-steps-dpoorpokto)
15. [iOS/macOS export notes (Core ML)](#iosmacos-export-notes-core-ml)
16. [Repro checklists](#repro-checklists)
17. [License & attribution](#license--attribution)

---

## Overview

This lane fine-tunes **Qwen-0.5B-Instruct** (or similar) with parameter-efficient methods:

- **LoRA** (baseline) and **DoRA** (weight decomposition variant).
- **KL-regularized SFT** to keep output close to the base policy (reduces drift / weird artifacts on tiny datasets).
- **ChatML** data (system/user/assistant messages) for Qwen-native prompting and cleaner generations.
- **Fast probes** (5–10 min) and **regular/long** runs controlled by YAML.

### Key features

- **Modular**: slim CLI + small `src/airoad/sft/*` helpers.
- **Nice logging**: live per-log prints + CSV/JSON artifacts + time metrics.
- **Early-stop**: plateau on training loss, optional eval-loss early stop.
- **Style nudges**: markdown slice + “negative tokens guard” data.

---

## What’s in this repo (LoRA lane)

**Slim CLI** (single entrypoint):

```

scripts/slm/qwen05b_lora_train.py        # or qwen05b_lora_train_slim.py in your branch

```

**Modular helpers:**

```

src/airoad/sft/data.py        # load JSONL, format ChatML/Alpaca → Qwen chat-template, HF datasets
src/airoad/sft/callbacks.py   # LossLogger, LivePrinter, TimeLimit, PlateauStop (early-stop)
src/airoad/sft/trainers.py    # KLSFTTrainer (TRL), KLTrainer (HF fallback), builders
src/airoad/sft/lora.py        # LoRA/DoRA target detection + wrapper
src/airoad/sft/**init**.py    # (optional) re-exports

```

**Data utilities (optional but recommended):**

```

scripts/data/build_markdown_slice.py   # add Markdown structure to a subset of answers
scripts/data/make_neg_guard.py         # small “avoid artifacts” set
scripts/data/pack_mix_chatml.py        # weighted mix + shuffle + dedupe

```

**Evaluation helpers:**

```

scripts/slm/qwen05b_compare.py         # Base vs tuned side-by-side sampling (ChatML)
scripts/eval/style_eval.py             # Markdown adherence, banned tokens, len, tok/s → JSON report
scripts/slm/qwen05b_lora_merge.py      # Merge LoRA adapters → single HF checkpoint

````

---

## Prereqs & Environment

- Apple Silicon macOS (M-series) or Linux with CUDA.
- Python 3.11+ (we use **uv** for package/env management).
- Torch w/ **MPS** (macOS) or CUDA (Linux), Transformers, TRL, PEFT, Datasets.

**Install (uv):**
```bash
uv python install 3.11
uv sync --python 3.11 --extra llm      # uses pyproject optional-deps "llm"
````

**Sanity (MPS):**

```bash
uv run python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
```

---

## Data formats & prep

### ChatML (preferred)

Each line is a JSON object:

```json
{"messages":[
  {"role":"system","content":"...voice..."},
  {"role":"user","content":"...prompt..."},
  {"role":"assistant","content":"...answer..."}
]}
```

### Alpaca triples (supported)

```json
{"instruction":"...", "input":"...", "output":"..."}
```

Alpaca rows are **auto-wrapped** into ChatML with your `system_prompt`.

### Build the mixed dataset (once)

```bash
# 1) add Markdown structure to ~60 items
uv run python scripts/data/build_markdown_slice.py \
  --infile data/sft/spirit_chatml.jsonl \
  --out    data/sft/spirit_chatml_md.jsonl \
  --limit  60

# 2) “negative tokens” guard
uv run python scripts/data/make_neg_guard.py \
  --out data/sft/spirit_chatml_neg.jsonl

# 3) Weighted mix 3:1:1 (core : md : neg)
uv run python scripts/data/pack_mix_chatml.py \
  --inputs data/sft/spirit_chatml.jsonl:3 data/sft/spirit_chatml_md.jsonl:1 data/sft/spirit_chatml_neg.jsonl:1 \
  --out    data/sft/spirit_chatml_mix.jsonl \
  --shuffle --seed 42 --dedupe
```

---

## Training — the slim CLI

**Fast probe (5–10 min):**

```bash
uv run python scripts/slm/qwen05b_lora_train.py \
  --config configs/slm/qwen05b_lora_fast_plain.yaml
```

Artifacts are written to `output_dir/`:

* **adapters** / tokenizer / config
* **train_log.csv** / **train_log.json**
* **time_metrics.json** (setup/train/total seconds, steps/s, tokens/s)

---

## Training flavors (configs)

Drop-in documented files (examples):

* `configs/slm/qwen05b_lora_fast_plain.yaml` – baseline LoRA SFT
* `configs/slm/qwen05b_lora_fast_dora.yaml` – DoRA only
* `configs/slm/qwen05b_lora_fast_kl.yaml` – KL-SFT only
* `configs/slm/qwen05b_lora_fast_dora_kl.yaml` – DoRA + KL (robust small-data)

Key fields:

* **Data**: `dataset_path`, `system_prompt`
* **Speed**: `max_steps`, `max_seq_len` (**biggest lever**), `grad_accum`, `ablate_examples`, `max_minutes`
* **LoRA**: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_mode` (`attn` | `attn_mlp` | `auto`), `use_dora`
* **KL**: `kl_lambda` (0 disables), `kl_tau` (1.0 typical)
* **Early-stop**: train-loss plateau knobs; optional eval-loss early stop
* **QoL**: `max_grad_norm`, `print_every_steps`

*Rule of thumb:* For fast probes use `max_seq_len: 512`, `max_steps: 60–120`, `ablate_examples: 96`.

---

## Merging adapters & inference

**Merge LoRA adapters → single HF checkpoint:**

```bash
uv run python scripts/slm/qwen05b_lora_merge.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapters outputs/qwen05b_fast_plain \
  --out checkpoints/qwen05b_fast_plain_merged
```

**Compare base vs tuned (adapters or merged):**

```bash
# adapters
uv run python scripts/slm/qwen05b_compare.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapters outputs/qwen05b_fast_plain

# merged
uv run python scripts/slm/qwen05b_compare.py \
  --base checkpoints/qwen05b_fast_plain_merged
```

---

## Evaluation & style scoring

**Style adherence (Markdown/banned tokens/len/tok/s):**

```bash
uv run python scripts/eval/style_eval.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapters outputs/qwen05b_fast_plain \
  --prompts data/sft/spirit_chatml.jsonl \
  --n 60 \
  --out outputs/style_eval_fast_plain.json
```

The evaluator prints a summary and saves full sample-level details.

We recommend tracking:

* `pct_list`, `pct_heading`: higher is better if you want Markdown style
* `avg_banned_hits`: should be near 0
* `avg_tokens_per_s`: throughput reference on your hardware

---

## Early stopping, time caps, and stability

* **Time cap**: `max_minutes` stops runs predictably (useful in fast sweeps).
* **Plateau stop**: halts when smoothed training loss doesn’t improve by `min_delta` across `patience` logs. Set a **warmup** floor via `early_stop_min_steps`.
* **Eval early stop**: if you have a dev set, flip `eval_early_stop: true` and provide `dev_path`.

---

## LoRA vs DoRA, and target placement

* **LoRA**: low-rank adapters for linear layers, training **~1–2%** of parameters.
* **DoRA**: weight-decomposed LoRA (supported in newer PEFT). Often steadier with similar capacity.

**Targets (`lora_target_mode`):**

* `attn` → q/k/v/(o) only (smaller adapters, faster).
* `attn_mlp` → attention + MLP proj (`gate/up/down`) (more steering power).
* `auto` → tries to detect (good default, we usually prefer `attn_mlp`).

*First pass:* `attn_mlp`, `r=16`, `alpha=32`, `dropout=0.05`.
*Speed-tightened:* `attn`, `r=8` for very fast probes.

---

## KL-regularized SFT

Add a small KL term to keep the student close to the base logits:

```
loss = CE(student) + kl_lambda * KL(teacher || student) * tau^2
```

* **When**: tiny datasets; you notice drift or artifacts (unwanted tags, verbosity).
* **How much**: start with `kl_lambda: 0.03–0.05`. Increase if style drifts; decrease if learning slows.
* **tau**: `1.0` is typical; larger smooths teacher distributions.

---

## Performance tips

* **Biggest lever: `max_seq_len`** (attention is O(L^2)). 512 is ~4× cheaper than 1024.
* **`grad_accum`**: smaller → faster wall time; larger → steadier grads. Use 8–16 for fast probes; 32 for regular runs.
* **`ablate_examples`**: 64–256 for quick iterations; `0` for full dataset.
* **LR**: slightly **higher** for fast probes (6e-5) to see a curve; **lower** with KL (5e-5 or less).
* **DoRA** and **KL** together are robust for small data; keep a **plain LoRA** baseline for comparison.

---

## Troubleshooting

### `TypeError: compute_loss() got an unexpected keyword argument 'num_items_in_batch'`

Recent TRL/Transformers call `compute_loss(..., num_items_in_batch=...)`.
**Fixed** in `src/airoad/sft/trainers.py` by accepting that kwarg in both custom trainers (KLSFTTrainer, KLTrainer).

### Ruff `E731` (“Do not assign a lambda expression, use a def”)

**Fixed** in `scripts/eval/style_eval.py` — replaced `lambda` with a local `def avg(key)`.

### Stray tags in generations (`Human:`, `<|user|>`, etc.)

* Ensure **ChatML formatting** throughout.
* Add **negative guard** data and mix it with small weight.
* Consider **KL-SFT** and/or **system_prompt** reminding “no literal chat tags”.

### MPS warnings (`pin_memory`)

We already disable `dataloader_pin_memory` where supported; harmless otherwise.

---

## Suggested “next steps” (DPO/ORPO/KTO)

Once SFT is stable:

* **DPO**: label small **chosen/rejected** pairs (RLAIF or human edits) and run a short preference train.
* **ORPO/KTO**: simpler preference variants (KTO can use “good only”).
* We recommend creating a new entrypoint (`scripts/slm/qwen05b_dpo_train.py`) that reuses the same modules.

---

## iOS/macOS export notes (Core ML)

The LoRA lane produces either **adapters** or a **merged** checkpoint:

* For **Core ML export**, merge adapters first (HF weights).
* Export with a ChatML-compatible prompt wrapper (enumerated context lengths 2k/4k).
* Let **Core ML** decide CPU/GPU/ANE at inference time (`computeUnits = .all`).

*(Exporter scripts are not included here yet — when you’re ready, we’ll add a separate `export_coreml.py` plus a small Swift loader.)*

---

## Repro checklists

### Fast probe (plain)

```bash
uv run python scripts/slm/qwen05b_lora_train.py --config configs/slm/qwen05b_lora_fast_plain.yaml
uv run python scripts/slm/qwen05b_compare.py --base Qwen/Qwen2.5-0.5B-Instruct --adapters outputs/qwen05b_fast_plain
uv run python scripts/eval/style_eval.py --base Qwen/Qwen2.5-0.5B-Instruct --adapters outputs/qwen05b_fast_plain --prompts data/sft/spirit_chatml.jsonl --n 60 --out outputs/style_eval_fast_plain.json
```

### Fast probe (DoRA + KL)

```bash
uv run python scripts/slm/qwen05b_lora_train.py --config configs/slm/qwen05b_lora_fast_dora_kl.yaml
uv run python scripts/slm/qwen05b_compare.py --base Qwen/Qwen2.5-0.5B-Instruct --adapters outputs/qwen05b_fast_dora_kl
uv run python scripts/eval/style_eval.py --base Qwen/Qwen2.5-0.5B-Instruct --adapters outputs/qwen05b_fast_dora_kl --prompts data/sft/spirit_chatml.jsonl --n 60 --out outputs/style_eval_fast_dora_kl.json
```

### Merge adapters (optional for deployment)

```bash
uv run python scripts/slm/qwen05b_lora_merge.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapters outputs/qwen05b_fast_dora_kl \
  --out checkpoints/qwen05b_fast_dora_kl_merged
```

---

## License & attribution

* **Qwen** models and tokenizers: see their respective licenses on Hugging Face.
* **This code**: use your repo’s LICENSE; attribute external libraries (Transformers, TRL, PEFT, Datasets).
* Please ensure your training data respects privacy and licensing.

---

## Appendix: Folder map

```
configs/
  slm/
    qwen05b_lora_fast_plain.yaml
    qwen05b_lora_fast_dora.yaml
    qwen05b_lora_fast_kl.yaml
    qwen05b_lora_fast_dora_kl.yaml

data/sft/
  spirit_chatml.jsonl
  spirit_chatml_md.jsonl
  spirit_chatml_neg.jsonl
  spirit_chatml_mix.jsonl

scripts/
  data/
    build_markdown_slice.py
    make_neg_guard.py
    pack_mix_chatml.py
  eval/
    style_eval.py
  slm/
    qwen05b_lora_train.py            # slim CLI (or qwen05b_lora_train_slim.py)
    qwen05b_lora_merge.py
    qwen05b_compare.py
    run_all_fast.sh                  # optional convenience runner

src/airoad/sft/
  __init__.py
  data.py
  callbacks.py
  trainers.py
  lora.py
```

---

**Questions or ideas?**
Open an issue or drop a note in the repo thread with your target device, data size, and goals (speed vs quality). We’ll tune the knobs or add the next trainer flavor (DPO/ORPO/KTO) to match.
