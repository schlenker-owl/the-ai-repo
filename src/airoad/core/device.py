from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def pick_device(user: Optional[str] = None) -> torch.device:
    """
    Choose a torch.device based on a user hint and availability.
    """
    if user:
        if user == "cpu":
            return torch.device("cpu")
        if user == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if (user.isdigit() or user == "cuda") and torch.cuda.is_available():
            return torch.device("cuda:0" if user == "cuda" else f"cuda:{user}")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """
    Deterministic seeds for Python, NumPy, and Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe if no cuda
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch >=2 uses deterministic algorithms when configured; we keep it fast by default.


def torch_dtype(name: str = "float32") -> torch.dtype:
    """
    Map simple strings to torch dtypes. (CPU-safe; fp16 only if non-CPU)
    """
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32
