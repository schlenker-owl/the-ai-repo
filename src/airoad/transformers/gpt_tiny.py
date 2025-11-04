# src/airoad/transformers/gpt_tiny.py
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Projections for q, k, v in one matmul
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask (broadcastable to (B, nH, T, T))
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        B, T, C = x.shape

        # Compute q, k, v and shape as (B, nH, T, Hd)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)  # (B,T,3,nH,Hd)
        q = qkv[:, :, 0].transpose(1, 2)  # (B,nH,T,Hd)
        k = qkv[:, :, 1].transpose(1, 2)  # (B,nH,T,Hd)
        v = qkv[:, :, 2].transpose(1, 2)  # (B,nH,T,Hd)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nH,T,T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B,nH,T,Hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        y = self.resid_drop(self.proj(y))  # (B,T,C)
        return y


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTiny(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 2,
        n_head: int = 2,
        n_embd: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) int64 token ids
        returns: logits (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block_size"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)

        x = self.tok(idx) + self.pos(pos)  # (B,T,C)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)  # (B,T,V)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively generate tokens, respecting block_size context window.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            next_id = torch.distributions.Categorical(logits=logits[:, -1, :]).sample().unsqueeze(1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
