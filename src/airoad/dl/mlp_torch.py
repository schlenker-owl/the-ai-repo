from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits
