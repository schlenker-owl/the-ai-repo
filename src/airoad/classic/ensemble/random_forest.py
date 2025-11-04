from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from airoad.classic.tree.tree_gini import DecisionTreeGini


@dataclass
class RandomForestClassifier:
    n_estimators: int = 25
    max_depth: int = 6
    min_samples_split: int = 2
    max_features: str | int | float = "sqrt"  # "sqrt", "log2", float in (0,1], or int
    bootstrap_ratio: float = 0.8
    random_state: Optional[int] = 0
    two_level_lookahead: bool = False

    # fitted
    trees_: List[DecisionTreeGini] | None = None
    feat_idx_: List[np.ndarray] | None = None
    n_classes_: int | None = None

    def _pick_feat_idx(self, d: int, rng: np.random.Generator) -> np.ndarray:
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                m = max(1, int(np.sqrt(d)))
            elif self.max_features == "log2":
                m = max(1, int(np.log2(d)))
            else:
                m = d
        elif isinstance(self.max_features, float):
            m = max(1, int(d * self.max_features))
        else:
            m = max(1, int(self.max_features))
        return rng.choice(d, size=m, replace=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        self.n_classes_ = int(np.max(y)) + 1
        rng = np.random.default_rng(self.random_state)
        self.trees_, self.feat_idx_ = [], []
        for _ in range(self.n_estimators):
            # bootstrap
            idx = rng.integers(0, n, size=int(self.bootstrap_ratio * n))
            feats = self._pick_feat_idx(d, rng)
            Xt, yt = X[idx][:, feats], y[idx]
            tree = DecisionTreeGini(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                two_level_lookahead=self.two_level_lookahead,
            ).fit(Xt, yt)
            self.trees_.append(tree)
            self.feat_idx_.append(feats)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        votes = np.zeros((n, self.n_classes_), dtype=np.int64)
        for tree, feats in zip(self.trees_, self.feat_idx_):
            pred = tree.predict(X[:, feats])
            for k in range(self.n_classes_):
                votes[:, k] += (pred == k).astype(np.int64)
        return votes.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).ravel()
        return float((self.predict(X) == y).mean())
