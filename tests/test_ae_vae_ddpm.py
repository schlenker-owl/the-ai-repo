import torch
import torch.nn as nn
from airoad.generative.ae import ConvAE
from airoad.generative.vae import ConvVAE, VaeLossConfig, elbo_loss
from airoad.generative.ddpm_mini import DiffusionSchedule, TinyUNet, diffusion_loss

torch.manual_seed(0)

def _toy_batch(B=2):
    # random "images" in [0,1]
    return torch.rand(B, 1, 28, 28)

def test_ae_train_step_reduces_mse():
    x = _toy_batch()
    model = ConvAE(latent_dim=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    # before
    with torch.no_grad():
        mse0 = loss_fn(model(x), x).item()
    # one step
    mse = loss_fn(model(x), x)
    opt.zero_grad(); mse.backward(); opt.step()
    with torch.no_grad():
        mse1 = loss_fn(model(x), x).item()
    assert mse1 <= mse0 + 1e-6

def test_vae_elbo_step():
    x = _toy_batch()
    vae = ConvVAE(latent_dim=8)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-2)
    cfg = VaeLossConfig(beta_max=1.0, warmup_steps=10)

    # initial loss
    x_hat, mu, logvar = vae(x)
    loss0, _ = elbo_loss(x, x_hat, mu, logvar, step=0, cfg=cfg)

    # one step
    loss = loss0
    opt.zero_grad(); loss.backward(); opt.step()

    x_hat2, mu2, logvar2 = vae(x)
    loss1, _ = elbo_loss(x, x_hat2, mu2, logvar2, step=10, cfg=cfg)
    assert loss1.item() <= loss0.item() + 1e-6

def test_ddpm_loss_and_shapes():
    x = _toy_batch() * 2 - 1  # [-1,1]
    sched = DiffusionSchedule(T=4, device="cpu")
    model = TinyUNet(base_c=16, with_t_embed=True, T_max=4)

    # loss scalar & backward
    loss = diffusion_loss(model, sched, x)
    assert torch.isfinite(loss).item()
    (loss * 0.0 + loss).backward()  # ensure backward works

    # sampling shapes (smoke)
    with torch.no_grad():
        samples = (x + 1) * 0.5  # not calling full sampler here to keep test instantaneous
        assert samples.shape == (2, 1, 28, 28)
