import typer

from airoad.classic.linear.linreg_numpy import LinearRegressionGD
from airoad.datasets.toy import make_linear_regression, standardize

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = 200,
    d: int = 1,
    noise: float = 0.1,
    lr: float = 0.1,
    epochs: int = 500,
    standardize_x: bool = True,
):
    X, y, w_true, b_true = make_linear_regression(n=n, d=d, noise=noise, seed=42)
    if standardize_x:
        X = standardize(X).X
    model = LinearRegressionGD(lr=lr, epochs=epochs, fit_intercept=True).fit(X, y)
    mse = model.mse(X, y)
    typer.echo(f"MSE: {mse:.6f}")
    typer.echo(
        f"True w[:3]: {w_true[:min(3,len(w_true))]}  |  Learned w[:3]: {model.w[:min(3,len(model.w))]}"
    )
    typer.echo(f"True b: {b_true:.3f} | Learned b: {model.b:.3f}")


if __name__ == "__main__":
    app()
