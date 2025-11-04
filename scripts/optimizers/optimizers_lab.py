import numpy as np
import typer

from airoad.optimizers.optimizers import SGD, Adam, Momentum, logistic_loss_grad

app = typer.Typer(add_completion=False)


def make_classification(n=500, d=6, margin=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d,))
    w = w / (np.linalg.norm(w) + 1e-12)
    b = rng.normal()
    logits = X @ w + b
    logits = logits + margin * np.sign(logits)  # increase separability
    y = (logits > 0).astype(np.float64)
    # standardize features
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y


def run(opt_name: str, steps: int, lr: float):
    X, y = make_classification()
    d = X.shape[1]
    w = np.zeros((d, 1))
    b = 0.0
    if opt_name == "sgd":
        opt = SGD(lr=lr)
    elif opt_name == "momentum":
        opt = Momentum(lr=lr)
    elif opt_name == "adam":
        opt = Adam(lr=lr)
    else:
        raise ValueError("opt_name ∈ {sgd,momentum,adam}")

    losses = []
    for _ in range(steps):
        loss, gw, gb = logistic_loss_grad(X, y.reshape(-1, 1), w, b, l2=0.0)
        w, b = opt.step(w, b, gw, gb)
        losses.append(loss)
    # accuracy
    p = 1.0 / (1.0 + np.exp(-(X @ w + b))).ravel()
    acc = float(((p >= 0.5).astype(np.int32) == y).mean())
    return losses[-1], acc


@app.command()
def main(steps: int = 300):
    res = {}
    res["sgd"] = run("sgd", steps=steps, lr=0.1)
    res["momentum"] = run("momentum", steps=steps, lr=0.05)
    res["adam"] = run("adam", steps=steps, lr=0.02)
    for k, (loss, acc) in res.items():
        typer.echo(f"{k:9s} -> loss={loss:.4f}  acc={acc:.3f}")
    # sanity expectation: adam ≤ momentum ≤ sgd on this tiny task (often true)


if __name__ == "__main__":
    app()
