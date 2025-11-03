import typer, numpy as np
from airoad.classic.ensemble.random_forest import RandomForestClassifier

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
    rf = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(X, y)
    acc = rf.accuracy(X, y)
    typer.echo(f"RandomForest acc={acc:.3f}")

if __name__ == "__main__":
    main()
