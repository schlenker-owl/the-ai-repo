
# Neural Networks & Transformers — Overview (v0.1)

This section of **the-ai-repo** (`airoad`) collects learn-by-doing labs for neural models:
- **MLP (PyTorch)**
- **CNN (PyTorch)**
- **Char RNN/GRU/LSTM (PyTorch)**
- **Tiny GPT-style Transformer (from scratch)**

Each model is designed to run fast on **Apple Silicon MPS or CPU**, teach core mechanics, and provide a path to deeper work.

## Contents
- [Quickstart](#quickstart)
- [What’s Implemented](#whats-implemented)
- [Recommended Reading Order](#recommended-reading-order)
- [Next Steps](#next-steps)

## Quickstart

```bash
# ensure env is ready
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
uv run python -m pip install -e .
uv run pytest -q

# quick demos (each ~minutes)
uv run python scripts/train_mlp_torch.py --steps 300                # XOR with MLP
uv run python scripts/train_cnn_mnist.py --steps 300                # MNIST subset
uv run python scripts/train_char_rnn.py --model lstm --steps 300    # char-level LSTM
uv run python scripts/train_gpt_tiny.py --steps 300                 # tiny GPT
````

## What’s Implemented

| Topic                       | Code                                                      | Script                       | Tests                               |
| --------------------------- | --------------------------------------------------------- | ---------------------------- | ----------------------------------- |
| **MLP (PyTorch)**           | `src/airoad/dl/mlp_torch.py`                              | `scripts/train_mlp_torch.py` | `tests/test_mlp_torch.py` (via XOR) |
| **CNN (PyTorch)**           | `src/airoad/dl/cnn_torch.py`                              | `scripts/train_cnn_mnist.py` | `tests/test_cnn_shapes.py`          |
| **Char RNN/GRU/LSTM**       | `src/airoad/dl/char_rnn.py`, `src/airoad/dl/char_data.py` | `scripts/train_char_rnn.py`  | `tests/test_char_models.py`         |
| **Tiny GPT (from scratch)** | `src/airoad/transformers/gpt_tiny.py`                     | `scripts/train_gpt_tiny.py`  | `tests/test_char_models.py`         |

## Recommended Reading Order

1. **MLP** → linear layers, nonlinearity, BCEWithLogitsLoss, optimizer loop.
2. **CNN** → convolutional inductive bias, pooling, shapes.
3. **RNN/GRU/LSTM** → sequence modeling, gating vs. vanishing gradients.
4. **Tiny GPT** → causal self-attention, masking, residuals, layer norm.

## Next Steps

* Add **schedulers** (warmup + cosine), **weight decay**, **gradient clipping**.
* Compare **teacher forcing** strategies for RNNs; try **scheduled sampling**.
* Extend GPT with **dropout/attn dropout sweeps**, **tied embeddings**, **top-k/nucleus sampling**.
* Add **Seq2Seq w/ attention**, and a **NumPy backprop MLP** to contrast with autograd.

👉 See detailed docs:

* `CNN_AND_MLP.md`
* `RNN_LSTM_GRU.md`
* `TRANSFORMER_GPT_TINY.md`
