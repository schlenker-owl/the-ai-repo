# scripts/train_gpt_tiny.py
import pathlib

import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from airoad.dl.char_data import CharDataset
from airoad.transformers.gpt_tiny import GPTTiny
from airoad.utils.device import pick_device

app = typer.Typer(add_completion=False)


def _read_text(path: pathlib.Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    # tiny fallback text to keep the demo self-contained
    return "To be, or not to be, that is the question.\n" * 500


@app.command()
def main(
    data_path: str = "data/tinyshakespeare.txt",
    steps: int = 300,
    batch_size: int = 32,
    block_size: int = 128,
    lr: float = 3e-3,
    n_layer: int = 2,
    n_head: int = 2,
    n_embd: int = 64,
    dropout: float = 0.1,
):
    text = _read_text(pathlib.Path(data_path))
    ds = CharDataset(text, block_size=block_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    dev = pick_device()
    model = GPTTiny(
        vocab_size=ds.vocab.size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(dev)

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

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            typer.echo(f"step {step:04d}  loss={loss.item():.4f}")

    # quick sample (avoid backslash in f-string expression)
    with torch.no_grad():
        prompt = torch.tensor([[ds.vocab.stoi[text[0]]]], device=dev)
        out_ids = model.generate(prompt, max_new_tokens=50).squeeze(0).tolist()
        itos = ds.vocab.itos
        sample = "".join(itos[i] for i in out_ids)
        clean = sample[:100].replace("\n", " ")
        typer.echo(f"sample: {clean} ...")


if __name__ == "__main__":
    app()
