# scripts/train_cnn_mnist.py
import typer, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from airoad.dl.cnn_torch import SimpleCNN
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)

@app.command()
def main(batch_size: int = 64, lr: float = 1e-3, steps: int = 500, limit_train: int = 5000, limit_test: int = 1000):
    dev = pick_device()
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    if limit_train and limit_train < len(train):
        train = Subset(train, range(limit_train))
    if limit_test and limit_test < len(test):
        test = Subset(test, range(limit_test))
    dl = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    tl = DataLoader(test, batch_size=256, shuffle=False)

    model = SimpleCNN(num_classes=10).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    it = iter(dl)
    model.train()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:04d} loss={loss.item():.4f}")

    # quick eval
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.numel()
    print(f"test acc ~ {correct/total:.3f}")

if __name__ == "__main__":
    app()
