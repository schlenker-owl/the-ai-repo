import typer, numpy as np
from airoad.classic.linear.ridge_lasso import ridge_closed_form, ridge_gd, lasso_coordinate_descent

app = typer.Typer(add_completion=False)

def make_regression(n: int = 200, d: int = 5, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d,))
    b_true = rng.normal()
    y = X @ w_true + b_true + noise * rng.normal(size=n)
    return X, y, w_true, b_true

@app.command()
def ridge(n: int = 200, d: int = 5, lam: float = 0.5, lr: float = 0.1, epochs: int = 300):
    X, y, w_true, b_true = make_regression(n, d)
    w_cf, b_cf = ridge_closed_form(X, y, lam, fit_intercept=True)
    w_gd, b_gd = ridge_gd(X, y, lam, lr=lr, epochs=epochs, fit_intercept=True)
    mse_cf = np.mean((X @ w_cf + b_cf - y) ** 2)
    mse_gd = np.mean((X @ w_gd + b_gd - y) ** 2)
    typer.echo(f"[ridge] λ={lam}  MSE(cf)={mse_cf:.6f}  MSE(gd)={mse_gd:.6f}")
    typer.echo(f"w_true[:3]={w_true[:3]} | w_cf[:3]={w_cf[:3]} | w_gd[:3]={w_gd[:3]}")
    typer.echo(f"b_true={b_true:.3f} | b_cf={b_cf:.3f} | b_gd={b_gd:.3f}")

@app.command()
def lasso(n: int = 200, d: int = 10, lam: float = 0.3):
    # sparse true weights to visualize shrinkage
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    w_true = np.zeros(d); w_true[: max(1, d // 5)] = rng.normal(size=max(1, d // 5))
    b_true = rng.normal()
    y = X @ w_true + b_true + 0.1 * rng.normal(size=n)
    w, b = lasso_coordinate_descent(X, y, lam, fit_intercept=True)
    k_true = int((w_true != 0).sum()); k_hat = int((np.abs(w) < 1e-8).sum())
    mse = np.mean((X @ w + b - y) ** 2)
    typer.echo(f"[lasso] λ={lam}  MSE={mse:.6f}  zeros={k_hat}/{len(w)} (true nonzeros={k_true})")
    typer.echo(f"w[:6]={w[:6]} | b={b:.3f}")

if __name__ == "__main__":
    app()
