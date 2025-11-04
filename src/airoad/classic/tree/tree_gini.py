# src/airoad/models/tree_gini.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------
# Impurity utilities
# ---------------------------


def _gini_from_counts(cnt: np.ndarray) -> float:
    """Gini impurity from class-counts vector."""
    n = cnt.sum()
    if n <= 0:
        return 0.0
    p = cnt / n
    return 1.0 - float((p * p).sum())


def _gini_labels(y: np.ndarray, n_classes: int) -> float:
    """Gini impurity from labels (ints in [0, n_classes))."""
    return _gini_from_counts(np.bincount(y, minlength=n_classes))


# ---------------------------
# Split search (1 feature)
# ---------------------------


def _best_split_feature(x: np.ndarray, y: np.ndarray, n_classes: int) -> Tuple[float, float] | None:
    """
    Best threshold (midpoint between sorted distinct values) for a single feature.
    Returns (threshold, gain) with gain = parent_gini - weighted_child_gini, or None if no improvement.
    """
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")  # stable for ties
    x_sorted = x[order]
    y_sorted = y[order]

    left_cnt = np.zeros((n_classes,), dtype=np.int64)
    total_cnt = np.bincount(y_sorted, minlength=n_classes)

    parent_imp = _gini_from_counts(total_cnt)
    best_gain = 0.0
    best_thr = None

    for i in range(1, n):
        c = y_sorted[i - 1]
        left_cnt[c] += 1

        # only consider thresholds between distinct consecutive values
        if x_sorted[i] == x_sorted[i - 1]:
            continue

        right_cnt = total_cnt - left_cnt
        n_left = i
        n_right = n - i
        g_left = _gini_from_counts(left_cnt)
        g_right = _gini_from_counts(right_cnt)
        child_imp = (n_left * g_left + n_right * g_right) / n
        gain = parent_imp - child_imp
        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (x_sorted[i] + x_sorted[i - 1])

    if best_thr is None:
        return None
    return (float(best_thr), float(best_gain))


def _best_split_any_feature(
    X: np.ndarray, y: np.ndarray, n_classes: int
) -> tuple[int, float, float] | None:
    """Best (feature_index, threshold, gain) across all features; None if no improvement."""
    best = None
    best_gain = 0.0
    for j in range(X.shape[1]):
        res = _best_split_feature(X[:, j], y, n_classes)
        if res is None:
            continue
        thr, gain = res
        if gain > best_gain:
            best_gain = gain
            best = (j, thr, gain)
    return best


# ---------------------------
# One-step lookahead helpers
# ---------------------------


def _min_impurity_after_one_split(X: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """
    Minimal weighted impurity achievable on (X,y) after **one** split
    (any single feature/threshold). If no split improves, return current impurity.
    """
    parent_imp = _gini_labels(y, n_classes)
    best = _best_split_any_feature(X, y, n_classes)
    if best is None:
        return parent_imp
    # gain = parent_imp - child_imp  => child_imp = parent_imp - gain
    _, _, gain = best
    return float(parent_imp - gain)


def _two_level_best_total_imp(
    X: np.ndarray, y: np.ndarray, n_classes: int, min_samples_split: int
) -> tuple[Optional[int], Optional[float], float]:
    """
    Evaluate all (feature, threshold) candidates at this node using a **two-level lookahead**:
    For each candidate first split, we compute the minimal child impurities achievable with ONE
    additional split inside each child, then combine them (weighted). Return:
      (best_feature, best_threshold, best_total_imp_after_two_levels)
    If no candidate valid split, returns (None, None, current_impurity).
    """
    n = X.shape[0]
    parent_imp = _gini_labels(y, n_classes)
    best_feat = None
    best_thr = None
    best_total_imp = parent_imp  # default: no improvement

    for j in range(X.shape[1]):
        x = X[:, j]
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y[order]

        # prefix counts
        left_cnt = np.zeros((n_classes,), dtype=np.int64)

        for i in range(1, n):
            c = y_sorted[i - 1]
            left_cnt[c] += 1
            if x_sorted[i] == x_sorted[i - 1]:
                continue

            thr = 0.5 * (x_sorted[i] + x_sorted[i - 1])
            idx_left = x <= thr
            nL = int(idx_left.sum())
            nR = n - nL
            if nL < min_samples_split or nR < min_samples_split:
                continue

            X_L, y_L = X[idx_left], y[idx_left]
            X_R, y_R = X[~idx_left], y[~idx_left]

            impL2 = _min_impurity_after_one_split(X_L, y_L, n_classes)
            impR2 = _min_impurity_after_one_split(X_R, y_R, n_classes)
            total_imp = (nL * impL2 + nR * impR2) / n

            if total_imp < best_total_imp:
                best_total_imp = float(total_imp)
                best_feat = j
                best_thr = float(thr)

    return best_feat, best_thr, float(best_total_imp)


# ---------------------------
# Tree
# ---------------------------


@dataclass
class _Node:
    is_leaf: bool
    pred: int
    feat: Optional[int] = None
    thr: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class DecisionTreeGini:
    """
    Tiny Decision Tree classifier using Gini impurity (binary or multi-class).

    It supports **two-level lookahead** to handle XOR-like structures where a purely
    greedy split at the root has zero (or near-zero) gain, but a depth-2 plan yields
    strong impurity reduction.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_gain: float = 1e-7,
        two_level_lookahead: bool = True,  # enabled by default
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.two_level_lookahead = two_level_lookahead

        self.n_classes: int = 0
        self.root: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        self.n_classes = int(np.max(y)) + 1
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        counts = np.bincount(y, minlength=self.n_classes)
        pred = int(np.argmax(counts))

        # stop if pure / small / depth limit
        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or counts.max() == X.shape[0]
        ):
            return _Node(is_leaf=True, pred=pred)

        parent_imp = _gini_labels(y, self.n_classes)

        # 1) Greedy best (one split)
        greedy = _best_split_any_feature(X, y, self.n_classes)
        greedy_gain = 0.0
        if greedy is not None:
            _, _, g_gain = greedy
            greedy_gain = float(g_gain)

        # 2) Two-level lookahead (if depth allows 2 more levels)
        two_feat = two_thr = None
        two_gain = 0.0
        if (
            self.two_level_lookahead
            and (depth <= self.max_depth - 2)
            and (X.shape[0] >= 2 * self.min_samples_split)
        ):
            bf, bt, best_total_imp = _two_level_best_total_imp(
                X, y, self.n_classes, self.min_samples_split
            )
            if bf is not None:
                two_feat, two_thr = bf, bt
                two_gain = float(parent_imp - best_total_imp)

        # 3) Choose the better of greedy vs lookahead
        chosen = None
        if two_gain > greedy_gain and two_gain >= self.min_gain:
            chosen = ("lookahead", two_feat, two_thr, two_gain)
        elif greedy_gain >= self.min_gain and greedy is not None:
            chosen = ("greedy", greedy[0], greedy[1], greedy_gain)

        if chosen is None:
            return _Node(is_leaf=True, pred=pred)

        _, feat, thr, _gain = chosen
        idx_left = X[:, feat] <= thr
        left = self._build(X[idx_left], y[idx_left], depth + 1)
        right = self._build(X[~idx_left], y[~idx_left], depth + 1)
        return _Node(is_leaf=False, pred=pred, feat=feat, thr=thr, left=left, right=right)

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while node and not node.is_leaf:
            assert node.feat is not None and node.thr is not None
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.pred if node else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x) for x in X], dtype=np.int64)
