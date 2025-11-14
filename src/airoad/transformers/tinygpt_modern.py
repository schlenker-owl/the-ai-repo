# src/airoad/transformers/tinygpt_modern.py
from __future__ import annotations

import math
from typing import Literal

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


# ========== FFN VARIANTS (unchanged) ==========


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3, bias: bool = False):
        super().__init__()
        d_ff = int(round(mult * d_model))
        self.up1 = nn.Linear(d_model, d_ff, bias=bias)
        self.up2 = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down(self.up1(x) * F.silu(self.up2(x)))


class GeLUMLP(nn.Module):
    def __init__(self, d_model: int, mult: float = 4.0, bias: bool = False):
        super().__init__()
        d_ff = int(round(mult * d_model))
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class GEGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3, bias: bool = False):
        super().__init__()
        d_ff = int(round(mult * d_model))
        self.up1 = nn.Linear(d_model, d_ff, bias=bias)
        self.up2 = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down(self.up1(x) * F.gelu(self.up2(x)))


class ReGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3, bias: bool = False):
        super().__init__()
        d_ff = int(round(mult * d_model))
        self.up1 = nn.Linear(d_model, d_ff, bias=bias)
        self.up2 = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down(self.up1(x) * F.relu(self.up2(x)))


class ConvGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        mult: float = 8 / 3,
        kernel_size: int = 5,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding"
        d_ff = int(round(mult * d_model))
        pad = dilation * (kernel_size // 2)
        self.dw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
            groups=d_model,
            bias=bias,
        )
        self.up1 = nn.Linear(d_model, d_ff, bias=bias)
        self.up2 = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.dw(y)
        y = y.transpose(1, 2)
        return self.down(self.up1(y) * F.silu(self.up2(y)))


class DenseMoE(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3, experts: int = 4, bias: bool = False):
        super().__init__()
        assert experts >= 2
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, mult=mult, bias=bias) for _ in range(experts)]
        )
        self.router = nn.Linear(d_model, experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.router(x), dim=-1)  # [B,T,E]
        outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B,T,d,E]
        return (outs * w.unsqueeze(-2)).sum(dim=-1)  # [B,T,d]


def make_ffn(
    ffn_type: Literal["swiglu", "gelu_mlp", "geglu", "reglu", "conv_glu", "moe"],
    d_model: int,
    mult: float,
    bias: bool = False,
    conv_kernel: int = 5,
    conv_dilation: int = 1,
    moe_experts: int = 4,
) -> nn.Module:
    if ffn_type == "swiglu":
        return SwiGLU(d_model, mult=mult, bias=bias)
    if ffn_type == "gelu_mlp":
        return GeLUMLP(d_model, mult=mult, bias=bias)
    if ffn_type == "geglu":
        return GEGLU(d_model, mult=mult, bias=bias)
    if ffn_type == "reglu":
        return ReGLU(d_model, mult=mult, bias=bias)
    if ffn_type == "conv_glu":
        return ConvGLU(
            d_model, mult=mult, kernel_size=conv_kernel, dilation=conv_dilation, bias=bias
        )
    if ffn_type == "moe":
        return DenseMoE(d_model, mult=mult, experts=moe_experts, bias=bias)
    raise ValueError(f"Unknown ffn_type: {ffn_type}")


# ========== ATTENTION (QKV fusion + RoPE cache) ==========


class RoPE:
    """
    Rotary position embeddings with simple per-device cache of cos/sin tables.
    """

    _cache = (
        {}
    )  # {(device, d_half): {"cos": Tensor[max_pos,d_half], "sin": Tensor[max_pos,d_half]}}

    def __init__(self, head_dim: int, base: float = 10000.0, device="cpu"):
        assert head_dim % 2 == 0
        self.d_half = head_dim // 2
        self.base = base
        self.inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    def to(self, device):
        self.inv = self.inv.to(device)
        return self

    @staticmethod
    def _key(device: torch.device, d_half: int):
        return (device.type, device.index if device.index is not None else -1, d_half)

    def _ensure_cache(self, device: torch.device, max_pos: int):
        key = self._key(device, self.d_half)
        slot = self._cache.get(key)
        if slot is None or slot["cos"].size(0) < max_pos:
            # build/extend tables
            positions = torch.arange(max_pos, device=device)
            ang = torch.outer(positions, self.inv)  # [max_pos, d_half]
            self._cache[key] = {"cos": ang.cos(), "sin": ang.sin()}

    def angles(self, positions: torch.Tensor):
        device = positions.device
        max_pos = int(positions.max().item()) + 1
        self._ensure_cache(device, max_pos)
        key = self._key(device, self.d_half)
        cos = self._cache[key]["cos"].index_select(0, positions)  # [T, d_half]
        sin = self._cache[key]["sin"].index_select(0, positions)  # [T, d_half]
        return cos, sin

    @staticmethod
    def _apply(x, cos, sin):
        # x: [B,H,T,D]; cos/sin: [1,1,T,D/2]
        even, odd = x[..., ::2], x[..., 1::2]
        xe = even * cos - odd * sin
        xo = even * sin + odd * cos
        out = torch.empty_like(x)
        out[..., ::2] = xe
        out[..., 1::2] = xo
        return out

    def apply_qk(self, q, k, qpos, kpos):
        cos_q, sin_q = self.angles(qpos)
        cos_k, sin_k = self.angles(kpos)
        cos_q = cos_q[None, None, :, :]
        sin_q = sin_q[None, None, :, :]
        cos_k = cos_k[None, None, :, :]
        sin_k = sin_k[None, None, :, :]
        return self._apply(q, cos_q, sin_q), self._apply(k, cos_k, sin_k)


class MultiHeadSelfAttention(nn.Module):
    """
    Decoder self-attention with GQA/MQA and fused QKV projection for fewer kernel launches.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_base=10000.0, bias=False):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads
        self.dh = d_model // n_heads

        # Fused QKV: out = [Q: H*Dh (=d_model)] + [K: HKV*Dh] + [V: HKV*Dh]
        self.qkv_proj = nn.Linear(d_model, d_model + 2 * (n_kv_heads * self.dh), bias=bias)
        self.o_proj = nn.Linear(n_heads * self.dh, d_model, bias=bias)

        self.rope = RoPE(self.dh, base=rope_base)

        # cached fallback causal mask (only used if SDPA is unavailable)
        self._fallback_mask = None  # [1,1,T,T] on demand

    def _get_fallback_mask(self, T: int, device: torch.device):
        if self._fallback_mask is None or self._fallback_mask.size(-1) < T:
            m = torch.triu(torch.ones(T, T, device=device), 1).bool()  # [T,T]
            self._fallback_mask = m[None, None, :, :]  # [1,1,T,T]
        return self._fallback_mask[:, :, :T, :T]

    def forward(self, x, start_pos=0, kv_cache=None):
        B, T, _ = x.shape
        H, HKV, Dh = self.n_heads, self.n_kv_heads, self.dh
        dev = x.device
        self.rope.to(dev)

        # Fused projection then split
        qkv = self.qkv_proj(x)  # [B,T, d_model + 2*(HKV*Dh)]
        q, k, v = torch.split(qkv, [H * Dh, HKV * Dh, HKV * Dh], dim=-1)

        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, HKV, Dh).transpose(1, 2)  # [B,HKV,T,Dh]
        v = v.view(B, T, HKV, Dh).transpose(1, 2)

        if kv_cache is None:
            pos = torch.arange(start_pos, start_pos + T, device=dev)
            q, k = self.rope.apply_qk(q, k, pos, pos)
            # expand K/V to all heads (GQA/MQA)
            k = k.repeat_interleave(self.kv_groups, dim=1)  # [B,H,T,Dh]
            v = v.repeat_interleave(self.kv_groups, dim=1)

            if hasattr(F, "scaled_dot_product_attention"):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                att = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
                mask = self._get_fallback_mask(T, dev)
                att = att.masked_fill(mask, float("-inf")).softmax(-1)
                y = att @ v

            y = y.transpose(1, 2).reshape(B, T, H * Dh)
            return self.o_proj(y), {"k": k.detach(), "v": v.detach(), "pos_offset": start_pos + T}

        else:
            past_k, past_v = kv_cache["k"], kv_cache["v"]  # [B,H,Tpast,Dh]
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
                y = att.softmax(-1) @ vcat

            y = y.transpose(1, 2).reshape(B, T, H * Dh)
            return self.o_proj(y), {
                "k": kcat.detach(),
                "v": vcat.detach(),
                "pos_offset": past_len + T,
            }


# ---- Decoder Block ----
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: float = 8 / 3,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        bias: bool = False,
        ffn_type: Literal["swiglu", "gelu_mlp", "geglu", "reglu", "conv_glu", "moe"] = "swiglu",
        conv_kernel: int = 5,
        conv_dilation: int = 1,
        moe_experts: int = 4,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, n_kv_heads, rope_base=rope_base, bias=bias
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = make_ffn(
            ffn_type=ffn_type,
            d_model=d_model,
            mult=ffn_mult,
            bias=bias,
            conv_kernel=conv_kernel,
            conv_dilation=conv_dilation,
            moe_experts=moe_experts,
        )
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
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_mult: float = 8 / 3,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        bias: bool = False,
        ffn_type: Literal["swiglu", "gelu_mlp", "geglu", "reglu", "conv_glu", "moe"] = "swiglu",
        conv_kernel: int = 5,
        conv_dilation: int = 1,
        moe_experts: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    ffn_mult,
                    rope_base,
                    dropout,
                    bias=bias,
                    ffn_type=ffn_type,
                    conv_kernel=conv_kernel,
                    conv_dilation=conv_dilation,
                    moe_experts=moe_experts,
                )
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
