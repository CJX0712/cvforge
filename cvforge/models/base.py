"""models/base — 分类器基类与通用工具。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cvforge.core.types import Prediction


class BaseClassifier:
    name = "base"

    def __init__(self) -> None:
        self._fitted = False
        self._classes = None

    def available(self) -> bool:
        return True

    def fit(self, X, y) -> "BaseClassifier":
        raise NotImplementedError

    def predict(self, X) -> Prediction:
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fit() before predict()")
