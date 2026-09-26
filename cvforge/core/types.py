"""core/types — 全局数据类型（dataclass 容器）。

约定：
- 图像为单通道灰度，shape (H, W)，值归一化到 [0, 1] float64。
- ImageDataset 为多样本带标签集合；评测语义：准确率/宏 F1 越高越好。
- BenchmarkResult.rows 为每行一个「特征@分类器」组合、列为指标的 dict 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ImageDataset:
    """带标签的图像集合（灰度）。"""

    images: np.ndarray           # (N, H, W) float64 in [0, 1]
    labels: np.ndarray           # (N,) int
    class_names: List[str]
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.images = np.asarray(self.images, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64).ravel()
        if self.images.ndim == 3:
            self.images = self.images[:, :, :, None] if False else self.images
        # 统一为 (N, H, W)
        if self.images.ndim == 4 and self.images.shape[-1] == 1:
            self.images = self.images[..., 0]
        self.class_names = list(self.class_names)

    @property
    def n_samples(self) -> int:
        return int(self.images.shape[0])

    @property
    def n_classes(self) -> int:
        return int(len(self.class_names))

    @property
    def img_shape(self):
        return tuple(self.images.shape[1:])


@dataclass
class Features:
    """提取后的特征矩阵。"""

    X: np.ndarray
    name: str = "features"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)


@dataclass
class Prediction:
    """分类预测结果。"""

    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.y_pred = np.asarray(self.y_pred, dtype=np.int64).ravel()
        if self.y_proba is not None:
            self.y_proba = np.asarray(self.y_proba, dtype=np.float64)


@dataclass
class MetricResult:
    """单条指标结果。"""

    name: str
    value: float
    model: str

    def __str__(self) -> str:
        return f"{self.model} {self.name}={self.value:.4f}"


@dataclass
class BenchmarkResult:
    """跨「特征@分类器」组合的基准结果。"""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows}

    def to_table(self, metrics_order: Optional[Sequence[str]] = None) -> str:
        """渲染等宽文本表（CLI 友好）。"""
        if not self.rows:
            return "(no results)"
        all_keys: set = set()
        for r in self.rows:
            for k in r:
                if k != "model":
                    all_keys.add(k)
        if metrics_order:
            ordered = [m for m in metrics_order if m in all_keys]
            ordered += [m for m in all_keys if m not in metrics_order]
        else:
            ordered = sorted(all_keys)
        header = f"{'model':<28}" + "".join(f"{m:>12}" for m in ordered)
        lines = [header, "-" * len(header)]
        for r in self.rows:
            line = f"{str(r.get('model', '')):<28}"
            for m in ordered:
                v = r.get(m, float("nan"))
                try:
                    line += f"{float(v):>12.4f}"
                except (TypeError, ValueError):
                    line += f"{str(v):>12}"
            lines.append(line)
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
