"""features/hog — HOG 特征（复用 scikit-image 工业标准纹理描述子）。

复用 scikit-image.feature.hog（BSD-3，纯 numpy/scipy，活跃维护），非自研。
available() 在 scikit-image 缺失时返回 False → benchmark 自动跳过，纯像素路径兜底。
"""

from __future__ import annotations

import numpy as np

from cvforge.core.types import Features, ImageDataset
from cvforge.features.base import BaseExtractor


class HOGExtractor(BaseExtractor):
    name = "HOG(skimage)"

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
    ) -> None:
        super().__init__()
        self.orientations = int(orientations)
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)

    def available(self) -> bool:
        try:
            import skimage.feature  # noqa: F401

            return True
        except Exception:
            return False

    def extract(self, dataset: ImageDataset) -> Features:
        from skimage.feature import hog

        batch = self._as_batch(dataset)
        feats = []
        for img in batch:
            v = hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                feature_vector=True,
            )
            feats.append(v.astype(np.float64))
        return Features(X=np.stack(feats, axis=0), name=self.name)
