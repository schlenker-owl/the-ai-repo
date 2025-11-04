import torch
import torch.nn as nn
import typer

from airoad.seq2seq.attn_seq2seq import AttnSeq2Seq, ToySeqConfig, toy_reverse_batch

app = typer.Typer(add_completion=False)


@app.command()
def main(steps: int = 200, B: int = 64, T: int = 10, vocab: int = 30, lr: float = 3e-3):
    dev = "cpu"
    cfg = ToySeqConfig(vocab_size=vocab, emb_dim=64, hidden=128, pad_id=0)
    model = AttnSeq2Seq(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)

    for step in range(1, steps + 1):
        x, x_lens, y_in, y_out, enc_mask = toy_reverse_batch(B=B, T=T, vocab=vocab, seed=step)
        x, x_lens, y_in, y_out, enc_mask = (
            x.to(dev),
            x_lens.to(dev),
            y_in.to(dev),
            y_out.to(dev),
            enc_mask.to(dev),
        )
        logits = model(x, x_lens, y_in, enc_mask)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y_out.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                pred = logits.argmax(-1)
                correct = ((pred == y_out) | (y_out == cfg.pad_id)).float().mean().item()
            typer.echo(f"step {step:04d}  loss={loss.item():.4f}  token_acc={correct:.3f}")


if __name__ == "__main__":
    main()
