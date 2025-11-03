from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ToySeqConfig:
    vocab_size: int = 30
    emb_dim: int = 64
    hidden: int = 128
    pad_id: int = 0

class Encoder(nn.Module):
    def __init__(self, cfg: ToySeqConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.rnn = nn.GRU(cfg.emb_dim, cfg.hidden, batch_first=True)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, h  # out: (B,T,H), h: (1,B,H)

class AttnDecoder(nn.Module):
    def __init__(self, cfg: ToySeqConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.rnn = nn.GRU(cfg.emb_dim + cfg.hidden, cfg.hidden, batch_first=True)
        self.attn = nn.Linear(cfg.hidden + cfg.hidden, cfg.hidden)
        self.v = nn.Linear(cfg.hidden, 1, bias=False)
        self.proj = nn.Linear(cfg.hidden, cfg.vocab_size)

    def _attend(self, dec_h, enc_out, mask):
        # dec_h: (B,1,H); enc_out: (B,T,H)
        B, T, H = enc_out.shape
        dec_exp = dec_h.expand(B, T, H)
        e = torch.tanh(self.attn(torch.cat([dec_exp, enc_out], dim=-1)))  # (B,T,H)
        scores = self.v(e).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = torch.softmax(scores, dim=-1)  # (B,T)
        ctx = torch.bmm(w.unsqueeze(1), enc_out).squeeze(1)  # (B,H)
        return ctx, w

    def forward(self, y_in, enc_out, h0, enc_mask):
        # y_in: (B,T) teacher forcing (input shifted right)
        B, T = y_in.shape
        logits = []
        h = h0
        for t in range(T):
            yt = y_in[:, t:t+1]
            emb = self.emb(yt)  # (B,1,E)
            ctx, _ = self._attend(h.transpose(0,1), enc_out, enc_mask)  # dec_h=(B,1,H)
            dec_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)
            out, h = self.rnn(dec_in, h)  # out: (B,1,H)
            logits.append(self.proj(out))  # list of (B,1,V)
        return torch.cat(logits, dim=1)  # (B,T,V)

class AttnSeq2Seq(nn.Module):
    def __init__(self, cfg: ToySeqConfig):
        super().__init__()
        self.enc = Encoder(cfg)
        self.dec = AttnDecoder(cfg)
        self.cfg = cfg

    def forward(self, x, x_lens, y_in, enc_mask):
        enc_out, h = self.enc(x, x_lens)
        logits = self.dec(y_in, enc_out, h, enc_mask)
        return logits

def toy_reverse_batch(B=32, T=10, vocab=30, pad_id=0, seed=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns x, x_lens, y_in, y_out for a toy reverse task.
    x: (B,T) with padding; x_lens: (B,)
    y_in: (B,T) shifted right input '<s>'==pad_id for simplicity
    y_out: (B,T) target reversed sequence
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    lens = rng.integers(low=3, high=T+1, size=B)
    x = np.full((B, T), pad_id, dtype=np.int64)
    y_out = np.full((B, T), pad_id, dtype=np.int64)
    for i, L in enumerate(lens):
        seq = rng.integers(1, vocab, size=L)
        x[i, :L] = seq
        y_out[i, :L] = seq[::-1]
    y_in = np.roll(y_out, shift=1, axis=1)
    y_in[:, 0] = pad_id
    enc_mask = (x != pad_id).astype(np.int64)
    return (torch.tensor(x), torch.tensor(lens), torch.tensor(y_in), torch.tensor(y_out), torch.tensor(enc_mask))
