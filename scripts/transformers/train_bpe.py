# scripts/train_bpe.py
import typer, pathlib
from airoad.tokenizers.bpe import BPETokenizer

app = typer.Typer(add_completion=False)

def _read_lines(path: pathlib.Path) -> list[str]:
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    return ["hello world", "tokenization with bpe", "bpe merges tokens </w>"]

@app.command()
def main(data_path: str = "data/tinyshakespeare.txt",
         vocab_size: int = 800,
         min_freq: int = 2,
         out_path: str = "outputs/bpe.json"):
    path = pathlib.Path(data_path)
    texts = _read_lines(path)
    tok = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq).train(texts)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    # demo round-trip
    s = "hello world hello"
    enc = tok.encode(s)
    dec = tok.decode(enc)
    typer.echo(f"Saved {out_path}. Roundtrip: '{s}' -> {enc[:10]} -> '{dec}'")

if __name__ == "__main__":
    app()
