# tests/test_char_models.py
import torch, torch.nn as nn
from airoad.dl.char_data import CharDataset
from airoad.dl.char_rnn import CharRNN
from airoad.transformers.gpt_tiny import GPTTiny

def _toy_text():
    return ("hello world " * 200).strip()

def _one_step_loss(model, ds):
    x, y = ds[0]
    x = x.unsqueeze(0)  # (1,T)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1).unsqueeze(0).view(-1))
    return loss

def test_char_rnn_forward_and_trainstep():
    ds = CharDataset(_toy_text(), block_size=64)
    model = CharRNN(vocab_size=ds.vocab.size, emb_dim=32, hidden=64, kind="gru")
    loss1 = _one_step_loss(model, ds)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.zero_grad(); loss1.backward(); opt.step()
    loss2 = _one_step_loss(model, ds)
    assert loss2.item() <= loss1.item() + 1e-6

def test_gpt_tiny_forward_shape():
    ds = CharDataset(_toy_text(), block_size=64)
    model = GPTTiny(vocab_size=ds.vocab.size, block_size=64, n_layer=1, n_head=2, n_embd=32)
    x, _ = ds[0]
    x = x.unsqueeze(0)
    logits = model(x)
    assert logits.shape == (1, x.size(1), ds.vocab.size)
