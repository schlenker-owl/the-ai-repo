# scripts/train_char_rnn.py
import typer, pathlib, torch, torch.nn as nn
from torch.utils.data import DataLoader
from airoad.dl.char_data import CharDataset
from airoad.dl.char_rnn import CharRNN
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)

def _read_text(path: pathlib.Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ("hello world " * 1000).strip()

@app.command()
def main(data_path: str = "data/tinyshakespeare.txt",
         model: str = "lstm",
         steps: int = 300,
         batch_size: int = 32,
         block_size: int = 128,
         lr: float = 1e-2,
         emb_dim: int = 64,
         hidden: int = 128):
    text = _read_text(pathlib.Path(data_path))
    ds = CharDataset(text, block_size=block_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dev = pick_device()
    model = CharRNN(vocab_size=ds.vocab.size, emb_dim=emb_dim, hidden=hidden, kind=model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    it = iter(dl)
    model.train()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        logits = model(x)                   # (B,T,V)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            typer.echo(f"step {step:04d}  loss={loss.item():.4f}")

if __name__ == "__main__":
    app()
