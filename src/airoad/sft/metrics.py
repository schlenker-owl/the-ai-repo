from __future__ import annotations

import re
import string
from typing import Iterable, List

_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(preds: Iterable[str], refs: Iterable[str]) -> float:
    preds_n = [_normalize_text(p) for p in preds]
    refs_n = [_normalize_text(r) for r in refs]
    correct = sum(int(p == r) for p, r in zip(preds_n, refs_n))
    total = max(1, len(preds_n))
    return correct / total


def _lcs_len(a_tokens: List[str], b_tokens: List[str]) -> int:
    """Classic LCS DP for ROUGE-L."""
    n, m = len(a_tokens), len(b_tokens)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 using whitespace tokens."""
    a = _normalize_text(pred).split()
    b = _normalize_text(ref).split()
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    prec = lcs / max(1, len(a))
    rec = lcs / max(1, len(b))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l(preds: Iterable[str], refs: Iterable[str]) -> float:
    vals = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
    return sum(vals) / max(1, len(vals))


def compute_metrics(preds: Iterable[str], refs: Iterable[str]) -> dict:
    preds_l = list(preds)
    refs_l = list(refs)
    return {
        "em": exact_match(preds_l, refs_l),
        "rougeL": rouge_l(preds_l, refs_l),
    }
