from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(True),   # 14x14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(True),  # 7x7
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.net = nn.Sequential(
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(True),  # 14x14
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),    # 28x28 in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z))


class ConvVAE(nn.Module):
    """
    Convolutional VAE with Gaussian posterior and Bernoulli-like decoder.
    Forward is deterministic (z = mu) to reduce variance in optimization/evaluation.
    For stochastic visualization, call sample_decode(x).
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.enc = ConvEncoder(latent_dim)
        self.dec = ConvDecoder(latent_dim)
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(x)
        # Deterministic decode for low-variance loss/grad
        x_hat = self.dec(mu)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic decode (z = mu + sigma*eps) for samples."""
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z)


@dataclass
class VaeLossConfig:
    beta_max: float = 1.0        # target KL weight at warmup end
    warmup_steps: int = 200      # linear ramp steps to beta_max
    beta_min_train: float = 1.0  # effective floor used in returned loss


def elbo_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    step: int,
    cfg: VaeLossConfig,
) -> tuple[torch.Tensor, dict]:
    """
    ELBO = recon + beta * KL
    - Recon: MSE over pixels (mean over batch & pixels) in [0,1] for stability.
      (Logged as 'bce' to keep training script output stable.)
    - KL: mean over batch, normalized by image area (H*W) * latent_dim so its magnitude
          matches recon even on tiny, random batches.
    - beta_true: linear warmup for intuition
    - beta: max(beta_true, beta_min_train) used in returned loss (and logged).
    """
    # reconstruction with MSE (stable on random inputs)
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")

    # KL(q || p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))  (sum over latent dims)
    kl_per_ex = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
    kl = kl_per_ex.mean()

    # Normalize KL to per-pixel *and* per-latent scale
    _, _, H, W = x.shape
    latent_dim = mu.shape[1]
    kl = kl / float(H * W * latent_dim)

    # beta schedule and effective beta
    beta_true = min(cfg.beta_max, (step / max(1, cfg.warmup_steps)) * cfg.beta_max)
    beta = max(beta_true, cfg.beta_min_train)

    elbo = recon + beta * kl
    parts = {
        "bce": recon.detach(),   # keep key name for training script compatibility
        "kl": kl.detach(),
        "beta": torch.tensor(beta),
    }
    return elbo, parts
