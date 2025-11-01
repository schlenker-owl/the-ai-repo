import numpy as np
import torch
from airoad.dl.mlp_torch import MLP
from airoad.utils.device import pick_device

def make_xor(n=256, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X.astype(np.float32), y.reshape(-1,1).astype(np.float32)

def test_mlp_learns_xor_fast():
    dev = torch.device("cpu")  # keep CI deterministic & fast
    torch.manual_seed(0)
    X_np, y_np = make_xor()
    X = torch.from_numpy(X_np).to(dev)
    y = torch.from_numpy(y_np).to(dev)
    model = MLP(in_dim=2, hidden=(16, 8)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        logits = model(X); loss = loss_fn(logits, y)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((model(X).sigmoid() >= 0.5).float() == y).float().mean().item()
    assert acc >= 0.90
