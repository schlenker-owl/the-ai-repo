from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """
    Tiny convolutional Autoencoder for 1×28×28 MNIST-like images.
    Encoder: 1 -> 16 -> 32 (downsample via stride-2)
    Bottleneck: 64 dims (linear)
    Decoder: mirrored conv-transpose
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 14x14
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 7x7
        )
        self.enc_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.dec_lin = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.Unflatten(1, (32, 7, 7)),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 14x14
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # 28x28, [0,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_lin(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.dec_lin(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
