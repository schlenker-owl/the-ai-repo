import torch
import torch.nn as nn

from airoad.seq2seq.attn_seq2seq import AttnSeq2Seq, ToySeqConfig, toy_reverse_batch


def test_seq2seq_one_step_improves_loss():
    cfg = ToySeqConfig()
    model = AttnSeq2Seq(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)

    x, x_lens, y_in, y_out, enc_mask = toy_reverse_batch(B=16, T=8, vocab=cfg.vocab_size, seed=0)
    logits = model(x, x_lens, y_in, enc_mask)
    loss0 = loss_fn(logits.reshape(-1, cfg.vocab_size), y_out.reshape(-1))

    opt.zero_grad()
    loss0.backward()
    opt.step()

    logits2 = model(x, x_lens, y_in, enc_mask)
    loss1 = loss_fn(logits2.reshape(-1, cfg.vocab_size), y_out.reshape(-1))
    assert loss1.item() <= loss0.item() + 1e-6
