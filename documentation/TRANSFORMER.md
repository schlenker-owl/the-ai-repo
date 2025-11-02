# Tiny GPT — Causal Self-Attention from Scratch (v0.1)

**Paths**
- Model: `src/airoad/transformers/gpt_tiny.py`
- Script: `scripts/train_gpt_tiny.py`
- Tests: `tests/test_char_models.py`

## Intuition
Transformers replace recurrence with **self-attention**: each token attends to **previous** tokens (causal mask) to build context-aware representations.

## Scaled Dot-Product Attention (causal)
For heads $h=1,\dots,H$ with key dimension $d_k$,
```math
\mathrm{Attn}(Q,K,V) \;=\; \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V,
````

where $Q,K,V \in \mathbb{R}^{T \times d_k}$ per head, and $M$ is the **causal mask** with $-\infty$ above the main diagonal to prevent looking ahead.

## Block

Each Transformer block applies:

1. LayerNorm
2. **Causal self-attention** + residual
3. LayerNorm
4. MLP (GELU, Linear) + residual

## Positional Encoding

We use **learnable** positional embeddings:

```math
X_{\text{input}} \;=\; \mathrm{TokenEmbed}(x) \;+\; \mathrm{PosEmbed}(t).
```

## Generation

At inference, feed the current context (last `block_size` tokens), compute logits, sample the next token, append, and repeat.

## Run

```bash
uv run python scripts/train_gpt_tiny.py --steps 300 --block-size 128 --n-layer 2 --n-head 2 --n-embd 64
```

## Learn-by-tweaking

* Increase `--n-layer` or `--n-embd` for capacity (watch speed/memory).
* Try **top-k** or **nucleus** sampling for generation (easy extension).
* Compare dropout `0.0` vs `0.1` to see regularization effects.

## Pitfalls

* **Mask shape mismatches** → ensure broadcast shape `(1,1,T,T)` and slice `:T, :T`.
* **Sequence length > block_size** → assert and/or crop the context window.

