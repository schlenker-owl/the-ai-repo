import typer
from airoad.datasets.toy import make_classification_2d, standardize
from airoad.models.logreg_numpy import LogisticRegressionGD

app = typer.Typer(add_completion=False)

@app.command()
def main(n: int = 300, margin: float = 0.5, lr: float = 0.2, epochs: int = 800, l2: float = 0.0, standardize_x: bool = True):
    X, y, w_true, b_true = make_classification_2d(n=n, margin=margin, seed=123)
    if standardize_x:
        X = standardize(X).X
    model = LogisticRegressionGD(lr=lr, epochs=epochs, fit_intercept=True, l2=l2).fit(X, y)
    acc = model.accuracy(X, y)
    typer.echo(f"Accuracy: {acc:.3f}")
    typer.echo(f"True w: {w_true} | Learned w[:2]: {model.w[:2]} | b: {model.b:.3f}")

if __name__ == "__main__":
    app()
