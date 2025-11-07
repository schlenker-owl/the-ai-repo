from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def simple_token_count(s: str) -> int:
    return max(1, len(s.split()))


def chunk_text(
    doc_id: str, text: str, max_tokens: int = 350, overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Simple whitespace/token chunker with sliding overlap.
    """
    toks = text.split()
    out: List[Chunk] = []
    i = 0
    cid = 0
    step = max(1, max_tokens - overlap_tokens)
    while i < len(toks):
        window = toks[i : i + max_tokens]
        if not window:
            break
        cid_str = f"{doc_id}_{cid:04d}"
        out.append(Chunk(doc_id=doc_id, chunk_id=cid_str, text=" ".join(window)))
        cid += 1
        i += step
    return out
