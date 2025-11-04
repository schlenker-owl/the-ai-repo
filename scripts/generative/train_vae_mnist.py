import torch
import typer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from airoad.generative.vae import ConvVAE, VaeLossConfig, elbo_loss
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)


@app.command()
def main(
    batch_size: int = 128,
    steps: int = 600,
    lr: float = 2e-3,
    latent_dim: int = 32,
    warmup_steps: int = 200,
    limit_train: int = 10000,
):
    dev = pick_device()
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    if limit_train and limit_train < len(ds):
        ds = Subset(ds, range(limit_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    vae = ConvVAE(latent_dim=latent_dim).to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    cfg = VaeLossConfig(beta_max=1.0, warmup_steps=warmup_steps)

    it = iter(dl)
    for step in range(1, steps + 1):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dl)
            x, _ = next(it)
        x = x.to(dev)

        x_hat, mu, logvar = vae(x)
        loss, parts = elbo_loss(x, x_hat, mu, logvar, step, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            bce = float(parts["bce"])
            kl = float(parts["kl"])
            beta = float(parts["beta"])
            typer.echo(
                f"step {step:04d}  elbo={loss.item():.5f}  bce={bce:.5f}  kl={kl:.5f}  beta={beta:.3f}"
            )


if __name__ == "__main__":
    app()
