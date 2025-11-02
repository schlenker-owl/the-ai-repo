
# MLP & CNN — Concepts, Math, and Labs (v0.1)

## MLP (PyTorch)

**Paths**
- Code: `src/airoad/dl/mlp_torch.py`
- Script: `scripts/train_mlp_torch.py`
- Test: `tests/test_mlp_torch.py` (XOR)

### Intuition
Stack linear layers with nonlinearities to learn non-linear functions. On XOR, a 2-layer MLP demonstrates how hidden units form curved decision boundaries.

### Forward
For hidden layers \(h^{(l)}\) with ReLU,
$$
h^{(1)}=\sigma(XW^{(1)}+b^{(1)}),\quad
h^{(2)}=\sigma(h^{(1)}W^{(2)}+b^{(2)}),\quad
\hat{y}=\text{MLP}(X)
$$
For binary labels we use **BCEWithLogitsLoss**:
$$
\mathcal{L} = \frac{1}{n}\sum_i \mathrm{BCEWithLogits}(\hat{y}_i, y_i)
$$

### Run
```bash
uv run python scripts/train_mlp_torch.py --steps 300
````

### Learn-by-tweaking

* Change hidden sizes `(16, 8)`, LR, steps; see accuracy shift.
* Add `weight_decay=1e-3` to AdamW; observe regularization.

---

## CNN (PyTorch)

**Paths**

* Code: `src/airoad/dl/cnn_torch.py`
* Script: `scripts/train_cnn_mnist.py`
* Test: `tests/test_cnn_shapes.py`

### Intuition

Convolutions learn local features invariant to shifts. Pooling reduces spatial resolution and increases receptive field.

### Convolution

For kernel (K) and input patch (X_{i,j}),
$$
(Y)*{i,j} = \sum*{u,v} K_{u,v}, X_{i+u,,j+v}
$$

### Run

```bash
uv run python scripts/train_cnn_mnist.py --steps 300 --limit-train 4000 --limit-test 1000
```

### Learn-by-tweaking

* Replace MaxPool with stride-2 conv; compare accuracy.
* Reduce filters (e.g., 8/16/32) for speed; note effect on performance.

### Pitfalls

* Normalization: MNIST is forgiving, but CIFAR-10 needs per-channel mean/std.
* Overfitting: watch the gap between train and test accuracy; add dropout or weight decay.

