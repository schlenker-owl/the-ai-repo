import numpy as np
import typer

from airoad.unsupervised.pca import PCA

app = typer.Typer(add_completion=False)


def make_correlated(n=400, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    A = rng.normal(size=(d, d))
    A = A @ A.T  # SPD-ish mixing
    X = X @ np.linalg.cholesky(A)  # correlate features
    return X


@app.command()
def main(n: int = 400, d: int = 5, k: int = 2, seed: int = 0):
    X = make_correlated(n=n, d=d, seed=seed)
    pca = PCA(n_components=k).fit(X)
    Z = pca.transform(X)
    Xr = pca.inverse_transform(Z)
    rec_mse = float(np.mean((X - Xr) ** 2))
    var_ratio_sum = float(pca.explained_variance_ratio_.sum())
    typer.echo(f"PCA: k={k}  variance_ratio_sum={var_ratio_sum:.3f}  recon_mse={rec_mse:.6f}")


if __name__ == "__main__":
    app()
