"""features/base — 特征提取器基类与通用工具。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cvforge.core.types import Features, ImageDataset


class BaseExtractor:
    name = "base"

    def __init__(self) -> None:
        self._fitted = False

    def available(self) -> bool:
        return True

    def extract(self, dataset: ImageDataset) -> Features:
        raise NotImplementedError

    def _as_batch(self, dataset: ImageDataset) -> np.ndarray:
        imgs = np.asarray(dataset.images, dtype=np.float64)
        if imgs.ndim == 3:
            return imgs
        raise ValueError("expected 3D image batch (N,H,W)")
