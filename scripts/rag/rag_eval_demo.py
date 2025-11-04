import typer

from airoad.rag.eval import evaluate_qa, retrieve_topk

app = typer.Typer(add_completion=False)

_CORPUS = [
    "Paris is the capital of France. It is known for the Eiffel Tower.",
    "Berlin is the capital of Germany.",
    "Ottawa is the capital city of Canada.",
    "The Moon orbits the Earth roughly every 27 days.",
]

_QA = [
    {"q": "What is the capital of France?", "ref": "Paris"},
    {"q": "Which city is the capital of Germany?", "ref": "Berlin"},
]


@app.command()
def main():
    preds = []
    for ex in _QA:
        idxs = retrieve_topk(ex["q"], _CORPUS, k=1)
        top_doc = _CORPUS[idxs[0]]
        # naive answer: first word before "is the capital"
        if " is the capital" in top_doc:
            pred = top_doc.split(" is the capital")[0]
        else:
            pred = top_doc.split()[0]
        preds.append({"question": ex["q"], "predicted": pred, "reference": ex["ref"]})
    m = evaluate_qa(preds)
    typer.echo(f"RAG demo metrics: {m}")


if __name__ == "__main__":
    main()
