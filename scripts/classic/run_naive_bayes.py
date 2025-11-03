# scripts/run_naive_bayes.py
import typer, numpy as np
from airoad.models.naive_bayes import GaussianNB, MultinomialNB

app = typer.Typer(add_completion=False)

def make_gaussian(n=600, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=0.8, size=(n//2, d))
    X1 = rng.normal(loc=+1.0, scale=0.8, size=(n//2, d))
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2), dtype=np.int64)
    return X, y

def make_multinomial_docs():
    docs = [
        "spam spam ham spam",
        "ham eggs toast",
        "spam offer deal spam",
        "ham salad fresh",
        "spam winner claim",
        "ham meeting schedule",
    ]
    y = np.array([1,0,1,0,1,0], dtype=np.int64)  # 1=spam, 0=ham
    # simple bag-of-words vectorizer
    vocab = {}
    for d in docs:
        for w in d.split():
            if w not in vocab: vocab[w] = len(vocab)
    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for i, d in enumerate(docs):
        for w in d.split():
            X[i, vocab[w]] += 1
    return X, y, vocab

@app.command()
def gaussian():
    X, y = make_gaussian()
    clf = GaussianNB().fit(X, y)
    acc = clf.accuracy(X, y)
    typer.echo(f"GaussianNB acc={acc:.3f}")

@app.command()
def multinomial(alpha: float = 1.0):
    X, y, vocab = make_multinomial_docs()
    clf = MultinomialNB(alpha=alpha).fit(X, y)
    acc = clf.accuracy(X, y)
    typer.echo(f"MultinomialNB acc={acc:.3f}  V={len(vocab)}")

if __name__ == "__main__":
    app()
