
# Char RNN / LSTM / GRU — Sequence Modeling (v0.1)

**Paths**
- Data & Vocab: `src/airoad/dl/char_data.py`
- Models: `src/airoad/dl/char_rnn.py`
- Script: `scripts/train_char_rnn.py`
- Tests: `tests/test_char_models.py`

## Intuition
Predict the **next character** given the previous context. RNN maintains a hidden state; **LSTM/GRU** add gating to reduce vanishing gradients and carry information longer.

## RNN Cell
Given input \(x_t\) and previous hidden \(h_{t-1}\),
$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\qquad
o_t = W_{ho} h_t + b_o
$$

## LSTM (gates)
With input, forget, output gates \(i_t,f_t,o_t\) and candidate \(\tilde{c}_t\),
$$
\begin{aligned}
i_t &= \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i),\\
f_t &= \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f),\\
o_t &= \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o),\\
\tilde{c}_t &= \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c),\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Loss: **token-level cross-entropy** over the sequence.

## Run
```bash
# data fallback auto-loads if file missing
uv run python scripts/train_char_rnn.py --model lstm --steps 300 --block-size 128
````

## Learn-by-tweaking

* Compare `--model rnn`, `--model gru`, `--model lstm`.
* Increase `block-size` (context length) and watch loss.
* Try LR sweeps (1e-3, 3e-3, 1e-2) and note stability.

## Pitfalls

* Exploding gradients → add `torch.nn.utils.clip_grad_norm_`.
* Long sequences on CPU/MPS → raise batch size gradually, monitor memory.

