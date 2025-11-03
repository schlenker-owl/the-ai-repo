import typer, numpy as np
from airoad.classic.neighbors.knn import KNNClassifier, KNNRegressor

app = typer.Typer(add_completion=False)

@app.command()
def clf():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(100, 2)) + np.array([+3, 0])
    X1 = rng.normal(size=(100, 2)) + np.array([-3, 0])
    X = np.vstack([X0, X1]); y = np.array([0]*100 + [1]*100)
    knn = KNNClassifier(k=5).fit(X, y)
    acc = knn.accuracy(X, y)
    typer.echo(f"KNN clf acc={acc:.3f}")

@app.command()
def reg():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 1))
    y = (X[:, 0] ** 2) + 0.1 * rng.normal(size=200)
    knnr = KNNRegressor(k=7).fit(X, y)
    mse = knnr.mse(X, y)
    typer.echo(f"KNN reg mse={mse:.4f}")

if __name__ == "__main__":
    app()
