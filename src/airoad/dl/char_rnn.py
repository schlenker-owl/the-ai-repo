# src/airoad/dl/char_rnn.py
from __future__ import annotations

import torch.nn as nn


class CharRNN(nn.Module):
    """
    Embedding -> {RNN/LSTM/GRU} -> Linear logits
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden: int = 128,
        num_layers: int = 1,
        kind: str = "lstm",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if kind == "rnn":
            self.core = nn.RNN(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        elif kind == "gru":
            self.core = nn.GRU(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        else:
            self.core = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        self.kind = kind
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        """
        x: (B, T) token ids
        returns logits: (B, T, vocab)
        """
        emb = self.embed(x)  # (B,T,E)
        out, _ = self.core(emb)  # (B,T,H)
        logits = self.proj(out)  # (B,T,V)
        return logits
