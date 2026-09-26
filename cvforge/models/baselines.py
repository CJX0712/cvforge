"""models/baselines — 弱基线分类器（Majority / Random）。

这些不是「自研模型」，而是分类评测的通用参照基线（与异常检测的 Naive 同构），
用于量化 SOTA 特征/分类器的相对增益。实现为标准教科书逻辑。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cvforge.core.types import Prediction
from cvforge.models.base import BaseClassifier


class MajorityBaseline(BaseClassifier):
    name = "Majority"

    def fit(self, X, y) -> "MajorityBaseline":
        y = np.asarray(y, dtype=np.int64).ravel()
        vals, counts = np.unique(y, return_counts=True)
        self._pred = int(vals[int(np.argmax(counts))])
        self._fitted = True
        return self

    def predict(self, X) -> Prediction:
        self._require_fitted()
        n = int(np.asarray(X, dtype=np.float64).shape[0])
        return Prediction(y_pred=np.full(n, self._pred, dtype=np.int64))


class RandomBaseline(BaseClassifier):
    name = "Random"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._rs = int(random_state)

    def fit(self, X, y) -> "RandomBaseline":
        y = np.asarray(y, dtype=np.int64).ravel()
        self._classes = np.unique(y)
        self._rng = np.random.default_rng(self._rs)
        self._fitted = True
        return self

    def predict(self, X) -> Prediction:
        self._require_fitted()
        n = int(np.asarray(X, dtype=np.float64).shape[0])
        yp = self._rng.integers(0, len(self._classes), size=n)
        return Prediction(y_pred=self._classes[yp].astype(np.int64))
