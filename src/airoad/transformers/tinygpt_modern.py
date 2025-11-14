# src/airoad/transformers/tinygpt_modern.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Tokenizer (byte-level) ----
class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)

    @staticmethod
    def decode(ids: torch.Tensor) -> str:
        return bytes(int(x) for x in ids.tolist()).decode("utf-8", errors="ignore")


# ---- RMSNorm ----
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# ---- SwiGLU FFN (defaults to ~2.67x width for param parity) ----
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3, bias: bool = False):
        super().__init__()
        d_ff = int(round(mult * d_model))
        self.up1 = nn.Linear(d_model, d_ff, bias=bias)  # value
        self.up2 = nn.Linear(d_model, d_ff, bias=bias)  # gate
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down(self.up1(x) * F.silu(self.up2(x)))


# ---- RoPE (rotary position embeddings) ----
class RoPE:
    def __init__(self, head_dim: int, base: float = 10000.0, device="cpu"):
        assert head_dim % 2 == 0
        self.inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    def to(self, device):
        self.inv = self.inv.to(device)
        return self

    def _angles(self, positions: torch.Tensor):
        return torch.outer(positions, self.inv)  # [T, D/2]

    @staticmethod
    def _apply(x, cos, sin):
        even, odd = x[..., ::2], x[..., 1::2]
        xe = even * cos - odd * sin
        xo = even * sin + odd * cos
        out = torch.zeros_like(x)
        out[..., ::2] = xe
        out[..., 1::2] = xo
        return out

    def apply_qk(self, q, k, qpos, kpos):
        qang = self._angles(qpos)
        kang = self._angles(kpos)
        cq, sq = qang.cos()[None, None, :, :], qang.sin()[None, None, :, :]
        ck, sk = kang.cos()[None, None, :, :], kang.sin()[None, None, :, :]
        return self._apply(q, cq, sq), self._apply(k, ck, sk)


# ---- Self-Attn (GQA/MQA capable) ----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_base=10000.0, bias=False):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.dh, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.dh, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.dh, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.dh, d_model, bias=bias)
        self.rope = RoPE(self.dh, base=rope_base)

    def forward(self, x, start_pos=0, kv_cache=None):
        B, T, D = x.shape
        H, HKV, Dh = self.n_heads, self.n_kv_heads, self.dh
        dev = x.device
        self.rope.to(dev)

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, HKV, Dh).transpose(1, 2)  # [B,HKV,T,Dh]
        v = self.v_proj(x).view(B, T, HKV, Dh).transpose(1, 2)

        if kv_cache is None:
            pos = torch.arange(start_pos, start_pos + T, device=dev)
            q, k = self.rope.apply_qk(q, k, pos, pos)
            k = k.repeat_interleave(self.kv_groups, dim=1)  # expand to H
            v = v.repeat_interleave(self.kv_groups, dim=1)
            if hasattr(F, "scaled_dot_product_attention"):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                att = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
                mask = torch.triu(torch.ones(T, T, device=dev), 1).bool()
                att = att.masked_fill(mask[None, None, :, :], float("-inf")).softmax(-1)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)
            return self.o_proj(y), {"k": k.detach(), "v": v.detach(), "pos_offset": start_pos + T}
        else:
            past_k, past_v = kv_cache["k"], kv_cache["v"]
            past_len = past_k.size(2)
            qpos = torch.arange(past_len, past_len + T, device=dev)
            kpos = qpos
            q, knew = self.rope.apply_qk(q, k, qpos, kpos)
            knew = knew.repeat_interleave(self.kv_groups, dim=1)
            vnew = v.repeat_interleave(self.kv_groups, dim=1)
            kcat = torch.cat([past_k, knew], dim=2)
            vcat = torch.cat([past_v, vnew], dim=2)
            if hasattr(F, "scaled_dot_product_attention"):
                y = F.scaled_dot_product_attention(q, kcat, vcat, is_causal=False)
            else:
                att = (q @ kcat.transpose(-1, -2)) / math.sqrt(Dh)
                y = (att.softmax(-1)) @ vcat
            y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)
            return self.o_proj(y), {
                "k": kcat.detach(),
                "v": vcat.detach(),
                "pos_offset": past_len + T,
            }


# ---- Decoder Block ----
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_mult=8 / 3,
        rope_base=10000.0,
        dropout=0.0,
        bias=False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, n_kv_heads, rope_base=rope_base, bias=bias
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, mult=ffn_mult, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, start_pos=0, kv_cache=None):
        y, new_cache = self.attn(self.norm1(x), start_pos=start_pos, kv_cache=kv_cache)
        x = x + self.drop(y)
        y = self.ffn(self.norm2(x))
        x = x + self.drop(y)
        return x, new_cache


# ---- TinyGPT (modern) ----
class TinyGPTModern(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=8 / 3,
        rope_base=10000.0,
        dropout=0.0,
        bias=False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, n_kv_heads, ffn_mult, rope_base, dropout, bias=bias)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache_list=None, start_pos=0):
        x = self.drop_in(self.embed(idx))
        caches = []
        for i, blk in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = blk(x, start_pos=start_pos, kv_cache=cache)
            caches.append(cache)
        x = self.norm_f(x)
        return self.lm_head(x), caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.9, top_k=50):
        self.eval()
        # device = idx.device
        caches = [None] * len(self.blocks)
        seq = idx
        for _ in range(max_new_tokens):
            logits, caches = self.forward(
                seq[:, -1:], kv_cache_list=caches, start_pos=seq.size(1) - 1
            )
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq
