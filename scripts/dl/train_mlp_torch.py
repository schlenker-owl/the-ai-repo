import numpy as np
import torch
import torch.nn as nn
import typer

from airoad.dl.mlp_torch import MLP
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)


def make_xor(n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32)  # XOR by sign
    # standardize
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


@app.command()
def main(steps: int = 300, lr: float = 1e-2, seed: int = 0):
    torch.manual_seed(seed)
    X_np, y_np = make_xor(n=512, seed=seed)
    dev = pick_device()
    X = torch.from_numpy(X_np).to(dev)
    y = torch.from_numpy(y_np).to(dev)

    model = MLP(in_dim=2, hidden=(32, 16)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(X)
        acc = ((logits.sigmoid() >= 0.5).float() == y).float().mean().item()
        typer.echo(f"Final loss: {loss.item():.4f}  Acc: {acc:.3f}  Device: {dev}")


if __name__ == "__main__":
    app()
