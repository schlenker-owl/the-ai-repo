from __future__ import annotations
from typing import List, Dict, Tuple

def exact_match(a: str, b: str) -> float:
    return float(a.strip().lower() == b.strip().lower())

def _lcs_len(x: List[str], y: List[str]) -> int:
    # classic DP; small strings here
    n, m = len(x), len(y)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        xi = x[i]
        for j in range(m):
            if xi == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l_f(pred: str, ref: str) -> float:
    # token-level (space) ROUGE-L F-score with beta=1
    ptoks = pred.strip().split()
    rtoks = ref.strip().split()
    if len(ptoks) == 0 or len(rtoks) == 0:
        return 0.0
    lcs = _lcs_len(ptoks, rtoks)
    prec = lcs / max(1, len(ptoks))
    rec  = lcs / max(1, len(rtoks))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def evaluate_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    pairs: list of (pred, ref)
    returns: {'exact_match': float, 'rougeL_f': float}
    """
    ems, rls = [], []
    for pred, ref in pairs:
        ems.append(exact_match(pred, ref))
        rls.append(rouge_l_f(pred, ref))
    import numpy as np
    return {
        "exact_match": float(np.mean(ems) if ems else 0.0),
        "rougeL_f": float(np.mean(rls) if rls else 0.0),
    }
