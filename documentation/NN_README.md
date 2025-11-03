# Neural Networks & Transformers — Overview (v0.2)

This section of **the-ai-repo** (`airoad`) collects learn-by-doing labs for neural models that run fast on **Apple Silicon (MPS)** or CPU and build intuition first:

- **MLP (PyTorch)**
- **CNN (PyTorch)**
- **Char RNN/GRU/LSTM (PyTorch)**
- **Seq2Seq with attention (PyTorch)**
- **Tiny GPT-style Transformer (from scratch)**

> For generative models (AE / VAE / DDPM-Mini), see **`docs/GENERATIVE.md`**.

---

## Contents
- [Quickstart](#quickstart)
- [What’s Implemented](#whats-implemented)
- [Recommended Reading Order](#recommended-reading-order)
- [Mini Math (RNN · Attention · LM Loss)](#mini-math-rnn--attention--lm-loss)
- [Next Steps](#next-steps)

---

## Quickstart

```bash
# env
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pytest -q

# quick demos (each finishes in minutes)
uv run python scripts/train_mlp_torch.py --steps 300                       # XOR with MLP
uv run python scripts/train_cnn_mnist.py --steps 300 --limit-train 4000    # MNIST subset (downloads)
uv run python scripts/train_char_rnn.py --model lstm --steps 300           # char-level LSTM
uv run python scripts/train_seq2seq_reverse.py --steps 200                 # toy reverse/copy with attention
uv run python scripts/train_gpt_tiny.py --steps 300                        # tiny GPT (char LM)
uv run python scripts/train_resnet18_transfer.py --steps 300 --limit-train 5000  # CIFAR-10 subset (downloads)
````

> **Notes**
>
> * MNIST/CIFAR will download on first run; we cap dataset sizes for speed.
> * On MPS you may see a benign `pin_memory` warning; it’s safe to ignore.

---

## What’s Implemented

| Topic                          | Code                                                      | Script                               | Tests                                  |
| ------------------------------ | --------------------------------------------------------- | ------------------------------------ | -------------------------------------- |
| **MLP (PyTorch)**              | `src/airoad/dl/mlp_torch.py`                              | `scripts/train_mlp_torch.py`         | `tests/test_mlp_torch.py` (XOR)        |
| **CNN (PyTorch)**              | `src/airoad/vision/cnn_torch.py`                          | `scripts/train_cnn_mnist.py`         | `tests/test_cnn_shapes.py`             |
| **Char RNN/GRU/LSTM**          | `src/airoad/dl/char_rnn.py`, `src/airoad/dl/char_data.py` | `scripts/train_char_rnn.py`          | `tests/test_char_models.py`            |
| **Seq2Seq + Attention**        | `src/airoad/seq2seq/attn_seq2seq.py`                      | `scripts/train_seq2seq_reverse.py`   | `tests/test_seq2seq_tiny.py`           |
| **Tiny GPT (from scratch)**    | `src/airoad/transformers/gpt_tiny.py`                     | `scripts/train_gpt_tiny.py`          | `tests/test_char_models.py`            |
| **Transfer learning (ResNet)** | `src/airoad/vision/transfer.py`                           | `scripts/train_resnet18_transfer.py` | `tests/test_resnet_transfer_shapes.py` |

> Generative labs (AE/VAE/DDPM-Mini) live in `src/airoad/generative/*` with demos under `scripts/generative/*` and their own tests. See **`docs/GENERATIVE.md`**.

---

## Recommended Reading Order

1. **MLP** → linear layers, nonlinearity, BCEWithLogitsLoss, optimizer loop.
2. **CNN** → convolutional inductive bias, pooling, tensor shapes.
3. **RNN/GRU/LSTM** → sequence modeling; gating vs. vanishing gradients.
4. **Seq2Seq + attention** → encoder/decoder with soft alignment.
5. **Tiny GPT** → causal self-attention, mask, residuals, layer norm.

> From here, jump to **Generative** (AE/VAE/DDPM-Mini) or **LLM Systemization** (LoRA SFT · RAG Eval · OpenAI-compatible server).

---

## Mini Math (RNN · Attention · LM Loss)

**RNN cell** (vanilla; LSTM/GRU add gates)

```math
h_t \;=\; \tanh\!\big(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\big),
\qquad
o_t \;=\; W_{ho}\,h_t + b_o.
```

**Bahdanau/Luong-style attention** (decoder step (t))

```math
e_{t,i} \;=\; v^\top \tanh\!\big(W_h\,h_t^{\text{dec}} \;+\; W_s\,h_i^{\text{enc}}\big),
\qquad
\alpha_{t,i} \;=\; \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})},
```

```math
c_t \;=\; \sum_i \alpha_{t,i}\,h_i^{\text{enc}},
\qquad
y_t \;=\; \mathrm{Decoder}\big(\,[\mathrm{Embed}(y_{t-1}) \,\|\, c_t],\, h_{t-1}^{\text{dec}}\big).
```

**Causal LM loss** (autoregressive next-token)

```math
\mathcal{L}_{\text{CausalLM}}
\;=\;
-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})\,.
```

---

## Next Steps

* **Training QoL:** add **schedulers** (warmup + cosine), **weight decay**, **grad clipping** toggles.
* **RNN training:** compare teacher forcing vs. scheduled sampling; monitor exposure bias.
* **GPT tiny:** add dropout sweeps, tied embeddings, and top-k / nucleus sampling.
* **Bridges:** implement a tiny **NumPy backprop MLP** to contrast with PyTorch autograd.
* **Vision:** try **transfer learning** variants (freeze/unfreeze backbone) and light augmentation.
* **See also:**

  * **`docs/GENERATIVE.md`** — AE · VAE · DDPM-Mini
  * **`docs/LLM_SYSTEMIZATION.md`** — LoRA SFT · RAG Eval · OpenAI-compatible server

