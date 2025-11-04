import numpy as np
import typer

from airoad.unsupervised.kmeans import KMeans

app = typer.Typer(add_completion=False)


def make_blobs(n=600, k=3, d=2, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * sep
    n_per = n // k
    Xs, ys = [], []
    for j in range(k):
        Xs.append(centers[j] + rng.normal(size=(n_per, d)))
        ys.append(np.full(n_per, j))
    return np.vstack(Xs), np.hstack(ys)


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Map each cluster -> majority true label, compute accuracy
    acc = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.any():
            vals, counts = np.unique(y_true[mask], return_counts=True)
            acc += counts.max()
    return float(acc / len(y_true))


@app.command()
def main(n: int = 600, k: int = 3, d: int = 2, sep: float = 4.0, seed: int = 0):
    X, y = make_blobs(n=n, k=k, d=d, sep=sep, seed=seed)
    km = KMeans(n_clusters=k, random_state=seed).fit(X)
    yhat = km.predict(X)
    p = purity(y, yhat)
    typer.echo(f"KMeans: inertia={km.inertia_:.2f}  purity={p:.3f}")


if __name__ == "__main__":
    app()
