from __future__ import annotations

from typing import List


def recall_at_k(golds: List[str], preds: List[List[str]], k: int = 5) -> float:
    """
    golds: list of gold doc ids (1 per query)
    preds: list of ranked candidate id lists per query
    """
    ok = 0
    for g, pr in zip(golds, preds):
        if g in pr[:k]:
            ok += 1
    return ok / max(1, len(golds))


def overlap_ratio(answer: str, context: str) -> float:
    """
    Crude faithfulness proxy: proportion of answer tokens found in context.
    """
    atoks = set(answer.lower().split())
    ctoks = set(context.lower().split())
    if not atoks:
        return 0.0
    return len(atoks & ctoks) / len(atoks)
