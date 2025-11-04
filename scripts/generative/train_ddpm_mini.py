import pathlib

import torch
import typer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision import utils as tvutils

from airoad.generative.ddpm_mini import DiffusionSchedule, TinyUNet, diffusion_loss, sample_loop
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)


@app.command()
def main(
    steps: int = 800,
    batch_size: int = 128,
    lr: float = 1e-3,
    T: int = 6,
    limit_train: int = 10000,
    out_path: str = "outputs/ddpm_grid.png",
):
    dev = pick_device()
    # Normalize to [-1,1] for diffusion
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    if limit_train and limit_train < len(ds):
        ds = Subset(ds, range(limit_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    sched = DiffusionSchedule(T=T, device=dev)
    model = TinyUNet(base_c=32, with_t_embed=True, T_max=T).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    it = iter(dl)
    for step in range(1, steps + 1):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dl)
            x, _ = next(it)
        x = x.to(dev)

        loss = diffusion_loss(model, sched, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            typer.echo(f"step {step:04d}  diff_loss={loss.item():.5f}")

    # sample a small grid
    with torch.no_grad():
        samples = sample_loop(model, sched, n=16, device=dev).cpu()
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tvutils.save_image(samples, out_path, nrow=4)
        typer.echo(f"Saved sample grid to: {out_path}")


if __name__ == "__main__":
    app()
