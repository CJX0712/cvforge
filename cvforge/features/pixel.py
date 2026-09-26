"""features/pixel — 像素展平特征（必选，零依赖，始终可用）。

将每张 (H,W) 灰度图展平为 H*W 维向量，作为最弱基线特征；
验证「SOTA 特征(HOG/ORB) 相对原始像素的增益」。
"""

from __future__ import annotations

import numpy as np

from cvforge.core.types import Features, ImageDataset
from cvforge.features.base import BaseExtractor


class PixelFlattenExtractor(BaseExtractor):
    name = "PixelFlatten"

    def available(self) -> bool:
        return True

    def extract(self, dataset: ImageDataset) -> Features:
        batch = self._as_batch(dataset)
        X = batch.reshape(batch.shape[0], -1)
        return Features(X=X, name=self.name)
