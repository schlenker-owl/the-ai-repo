# src/airoad/dl/char_data.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CharVocab:
    stoi: dict
    itos: list

    @property
    def size(self) -> int:
        return len(self.itos)


def build_vocab_from_text(text: str) -> CharVocab:
    chars = sorted(set(text))
    itos = list(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return CharVocab(stoi=stoi, itos=itos)


class CharDataset(Dataset):
    """
    Character-level next-token dataset.
    Returns (input_ids [T], target_ids [T]) pairs.
    """

    def __init__(self, text: str, block_size: int = 128):
        if len(text) < block_size + 2:
            text = (text + "\n") * ((block_size + 2) // max(1, len(text)) + 1)
        self.vocab = build_vocab_from_text(text)
        self.block_size = block_size
        self.data = np.array([self.vocab.stoi[ch] for ch in text], dtype=np.int64)

    def __len__(self):
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
