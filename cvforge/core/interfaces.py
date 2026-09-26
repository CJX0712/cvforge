"""interfaces — 解耦用的 Protocol 契约（依赖倒置，便于独立测试与替换实现）。

- FeatureExtractor: 统一特征提取接口；`available()` 支持可选后端离线降级。
- Classifier: 统一分类接口（fit/predict/predict_proba）。
- DataSource: 数据来源（合成 / 图像目录 / CSV）统一抽象。
- Metric: 评测指标。
- Evaluator: 对单个分类器在特征上给出指标结果。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from cvforge.core.types import Features, ImageDataset, Prediction


@runtime_checkable
class FeatureExtractor(Protocol):
    name: str

    def available(self) -> bool: ...

    def extract(self, images: ImageDataset) -> Features: ...


@runtime_checkable
class Classifier(Protocol):
    name: str

    def available(self) -> bool: ...

    def fit(self, X, y) -> "Classifier": ...

    def predict(self, X) -> Prediction: ...


@runtime_checkable
class DataSource(Protocol):
    def load(self) -> ImageDataset: ...


@runtime_checkable
class Metric(Protocol):
    name: str

    def evaluate(self, actual, pred) -> float: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, clf, X, y, test_size: float, random_state: int) -> List: ...
