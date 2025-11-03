import typer, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from airoad.generative.ae import ConvAE
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)

@app.command()
def main(
    batch_size: int = 128,
    steps: int = 500,
    lr: float = 1e-3,
    latent_dim: int = 64,
    limit_train: int = 10000,
):
    dev = pick_device()
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    if limit_train and limit_train < len(ds):
        ds = Subset(ds, range(limit_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ConvAE(latent_dim=latent_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    it = iter(dl)
    for step in range(1, steps + 1):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dl); x, _ = next(it)
        x = x.to(dev)
        x_hat = model(x)
        loss = loss_fn(x_hat, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            typer.echo(f"step {step:04d}  recon_mse={loss.item():.5f}")

if __name__ == "__main__":
    app()
