# scripts/run_softmax.py
import typer, numpy as np
from airoad.models.softmax_numpy import SoftmaxRegressionGD

app = typer.Typer(add_completion=False)

def make_multiclass(n=600, k=3, d=4, margin=0.8, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, k))
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    b = rng.normal(size=(k,))
    X = rng.normal(size=(n, d))
    logits = X @ W + b
    logits = logits + margin * np.sign(logits)
    y = logits.argmax(axis=1).astype(np.int64)
    # standardize
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X, y

@app.command()
def main(n:int=600, k:int=3, d:int=4, lr:float=0.1, epochs:int=500, l2:float=0.0):
    X, y = make_multiclass(n=n, k=k, d=d)
    clf = SoftmaxRegressionGD(lr=lr, epochs=epochs, l2=l2).fit(X, y)
    acc = clf.accuracy(X, y)
    typer.echo(f"Softmax: acc={acc:.3f}")

if __name__ == "__main__":
    app()
