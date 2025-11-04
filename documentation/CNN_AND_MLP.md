# MLP & CNN — Concepts, Math, and Labs (v0.2)

This page summarizes the **MLP** and **CNN** labs in `airoad` with quick math, paths, and runnable commands. Both are tuned for **Apple Silicon (MPS)** or CPU and finish in minutes.

---

## MLP (PyTorch)

**Paths**
- Code: `src/airoad/dl/mlp_torch.py`
- Script: `scripts/train_mlp_torch.py`
- Test: `tests/test_mlp_torch.py` (XOR)

### Intuition
An MLP stacks linear layers with nonlinearities to learn non-linear decision boundaries. On XOR, a 2-layer MLP cleanly demonstrates how hidden units bend the boundary.

### Forward
For hidden layers $h^{(l)}$ with ReLU,
```math
h^{(1)}=\sigma(XW^{(1)}+b^{(1)}),\quad
h^{(2)}=\sigma(h^{(1)}W^{(2)}+b^{(2)}),\quad
\hat{y}=\mathrm{MLP}(X).
````

For binary labels we use **BCEWithLogitsLoss**:

```math
\mathcal{L} \;=\; \frac{1}{n}\sum_{i=1}^{n}\mathrm{BCEWithLogits}(\hat{y}_i,\,y_i).
```

### Run

```bash
uv run python scripts/train_mlp_torch.py --steps 300
```

### Learn-by-tweaking

* Change hidden sizes `(16, 8)`, LR, steps; see accuracy shift.
* Add `weight_decay=1e-3` to AdamW; observe regularization.
* Try different activations (e.g., GELU, Tanh) and note training stability.

---

## CNN (PyTorch)

**Paths**

* Code: `src/airoad/vision/cnn_torch.py`
* Script: `scripts/train_cnn_mnist.py`
* Test: `tests/test_cnn_shapes.py`

### Intuition

Convolutions learn **local features** that are translation-tolerant. Pooling reduces spatial resolution and increases **receptive field** depth.

### Convolution

For kernel $K$ and input patch $X_{i,j}$,

```math
(Y)_{i,j} \;=\; \sum_{u,v} K_{u,v}\, X_{i+u,\,j+v}.
```

### Run

```bash
uv run python scripts/train_cnn_mnist.py --steps 300 --limit-train 4000 --limit-test 1000
```

### Learn-by-tweaking

* Replace `MaxPool2d(2)` with a stride-2 conv; compare accuracy/throughput.
* Reduce filters (e.g., 8/16/32) for speed; note the effect on performance.
* Add light augmentation (random crop/flip) to see gains on test accuracy.

### Pitfalls

* **Normalization:** MNIST is forgiving; CIFAR-10 needs per-channel mean/std and often stronger augmentation.
* **Overfitting:** watch train vs. test; add dropout or weight decay when gaps appear.
* **Shapes:** verify `(B, C, H, W)` ordering before convs; mismatches are common sources of bugs.

---

## Related docs

* `docs/NN_README.md` — overview of all neural labs (MLP, CNN, RNN/LSTM/GRU, Seq2Seq, GPT tiny).
* `docs/RNN_LSTM_GRU.md` — sequence models (char-level).
* `docs/TRANSFORMER_GPT_TINY.md` — causal self-attention, masking, residuals, layer norm.
* `docs/GENERATIVE.md` — AE · VAE · DDPM-Mini labs.
