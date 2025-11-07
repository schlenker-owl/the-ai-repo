from pathlib import Path

import numpy as np
import torch

from airoad.core.device import pick_device, seed_everything
from airoad.core.io import ensure_dir, load_json, save_json


def test_device_and_seed():
    d = pick_device(None)
    assert isinstance(d, torch.device)
    seed_everything(123)
    # Just verify entropy source is usable and deterministic hook doesn't crash.
    assert isinstance(np.random.RandomState(123).rand(), float)


def test_io_roundtrip(tmp_path: Path):
    p = tmp_path / "x" / "y.json"
    ensure_dir(p.parent)
    save_json(p, {"a": 1})
    obj = load_json(p)
    assert obj["a"] == 1
