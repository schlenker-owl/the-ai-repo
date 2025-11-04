# Char RNN / LSTM / GRU — Sequence Modeling (v0.2)

**Paths**
- Data & Vocab: `src/airoad/dl/char_data.py`
- Models: `src/airoad/dl/char_rnn.py`
- Script: `scripts/train_char_rnn.py`
- Tests: `tests/test_char_models.py`

---

## Intuition
We model the **next character** given the previous context. A vanilla RNN maintains a hidden state; **LSTM** and **GRU** introduce **gates** to control information flow, reduce vanishing gradients, and carry information longer.

---

## RNN Cell
Given input $x_t$ and previous hidden $h_{t-1}$,
```math
h_t \;=\; \tanh\!\big(W_{xh}x_t + W_{hh}h_{t-1} + b_h\big),
\qquad
o_t \;=\; W_{ho}\, h_t + b_o.
````

---

## LSTM (gates)

With input/forget/output gates $i_t,f_t,o_t$ and candidate $\tilde{c}_t$,

```math
\begin{aligned}
i_t &= \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i),\\
f_t &= \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f),\\
o_t &= \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o),\\
\tilde{c}_t &= \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c),\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
```

**Loss:** token-level cross-entropy over the sequence (teacher forcing).

---

## Run

```bash
# data fallback auto-loads if the file is missing
uv run python scripts/train_char_rnn.py --model lstm --steps 300 --block-size 128
```

---

## Learn-by-tweaking

* Compare `--model rnn`, `--model gru`, `--model lstm`.
* Increase `--block-size` (context length) and observe loss/quality.
* Try LR sweeps (`1e-3`, `3e-3`, `1e-2`) and note stability.
* Change embedding and hidden sizes; watch compute vs. quality.

---

## Pitfalls

* **Exploding gradients** → add clipping:

  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  ```
* **Long sequences on CPU/MPS** → grow batch/bptt gradually; monitor memory.
* **Data mixing** → shuffle character streams to avoid overfitting trivial patterns.

---

## Related labs

* `docs/TRANSFORMER_GPT_TINY.md` — causal self-attention (Tiny GPT)
* `docs/NN_README.md` — overview of neural labs (MLP/CNN/Seq2Seq/GPT)
* `docs/GENERATIVE.md` — AE · VAE · DDPM-Mini
