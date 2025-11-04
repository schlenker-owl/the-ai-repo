import numpy as np
import typer

from airoad.unsupervised.gmm import GMM, normalized_mutual_info

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


@app.command()
def main():
    X, y = make_blobs()
    gmm = GMM(n_components=3).fit(X)
    pred = gmm.predict(X)
    nmi = normalized_mutual_info(pred, y)
    typer.echo(f"GMM NMI={nmi:.3f}")


if __name__ == "__main__":
    main()
