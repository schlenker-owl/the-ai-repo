from __future__ import annotations
import math
import torch
import torch.nn as nn


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DiffusionSchedule:
    def __init__(self, T: int = 6, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | torch.device = "cpu"):
        self.T = T
        self.beta = make_beta_schedule(T, beta_start, beta_end).to(device)                     # (T,)
        self.alpha = 1.0 - self.beta                                                           # (T,)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)                                      # (T,)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise


# --- Tiny UNet-like backbone (very small for speed) ---

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.SiLU(),
        )

    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, c): super().__init__(); self.pool = nn.MaxPool2d(2); self.block = ConvBlock(c, c*2)
    def forward(self, x): return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, c): super().__init__(); self.up = nn.ConvTranspose2d(c, c//2, 2, stride=2); self.block = ConvBlock(c, c//2)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TinyUNet(nn.Module):
    """
    Predicts noise given noised image x_t and (optional) timestep embedding.
    Channels kept very small for speed.
    """
    def __init__(self, base_c: int = 32, with_t_embed: bool = True, T_max: int = 1000):
        super().__init__()
        self.with_t_embed = with_t_embed
        self.t_embed = nn.Embedding(T_max, base_c) if with_t_embed else None

        self.in_block = ConvBlock(1, base_c)        # 28x28 -> base
        self.down1 = Down(base_c)                   # base -> 2*base, 14x14
        self.bot   = ConvBlock(2*base_c, 2*base_c)
        self.up1   = Up(2*base_c)                   # 14x14 -> 28x28
        self.out   = nn.Conv2d(base_c, 1, 1)

    def forward(self, x, t: torch.Tensor | None = None):
        if self.with_t_embed and t is not None:
            # broadcast add a simple timestep embedding
            te = self.t_embed(t).view(-1, self.t_embed.embedding_dim, 1, 1)
            x = x + te[:, :1]  # inject a very small piece (single channel portion) for speed

        s1 = self.in_block(x)          # (B, base, 28, 28)
        d1 = self.down1(s1)            # (B, 2*base, 14, 14)
        b  = self.bot(d1)              # (B, 2*base, 14, 14)
        u1 = self.up1(b, s1)           # (B, base, 28, 28)
        out = self.out(u1)             # (B, 1, 28, 28)
        return out


# --- Training loss (predict noise) ---

def diffusion_loss(model: nn.Module, sched: DiffusionSchedule, x0: torch.Tensor) -> torch.Tensor:
    B = x0.size(0)
    t = torch.randint(low=0, high=sched.T, size=(B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    pred = model(x_t, t)
    return nn.functional.mse_loss(pred, noise)


@torch.no_grad()
def sample_loop(model: nn.Module, sched: DiffusionSchedule, n: int = 16, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Sample from pure noise with DDPM reverse (tiny, deterministic w/o classifier-free guidance).
    Returns a batch (n,1,28,28) in [0,1] after simple clamping.
    """
    x = torch.randn(n, 1, 28, 28, device=device)
    for t_inv in range(sched.T - 1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps = model(x, t)
        beta_t = sched.beta[t].view(-1, 1, 1, 1)
        alpha_t = sched.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = sched.alpha_bar[t].view(-1, 1, 1, 1)
        # DDPM step (DDIM-free, simple form)
        x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps)
        if t_inv > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = x + sigma * z
    x = (x.clamp(-1, 1) + 1) * 0.5  # map to [0,1]
    return x
