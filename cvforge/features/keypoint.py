"""features/keypoint — 关键点描述子（复用 OpenCV ORB / SIFT，可选后端）。

复用 OpenCV(cv2) 的 ORB / SIFT 工业标准局部特征（Apache-2，活跃维护），非自研。
每张图取描述子均值得到固定长度向量（ORB=32 / SIFT=128），无关键点则为零向量。
available() 在 cv2 缺失时返回 False → benchmark 自动跳过。
"""

from __future__ import annotations

import numpy as np

from cvforge.core.types import Features, ImageDataset
from cvforge.features.base import BaseExtractor


class _MeanDescriptorExtractor(BaseExtractor):
    _kind = "orb"  # 'orb' | 'sift'
    _desc_len = 32

    def __init__(self) -> None:
        super().__init__()
        self.name = f"{self._kind.upper()}(cv2)"

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401

            return True
        except Exception:
            return False

    def _detector(self):
        import cv2

        if self._kind == "orb":
            # 小尺寸平滑合成图上默认 FAST 阈值过高会漏掉全部关键点；
            # 放宽 fastThreshold/edgeThreshold 才能在形状边界取到关键点。
            return cv2.ORB_create(nfeatures=200, fastThreshold=0, edgeThreshold=0)
        return cv2.SIFT_create()

    def extract(self, dataset: ImageDataset) -> Features:
        batch = self._as_batch(dataset)
        feats = []
        for img in batch:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            det = self._detector()
            try:
                _, des = det.detectAndCompute(u8, None)
            except Exception:
                des = None
            if des is None or len(des) == 0:
                feats.append(np.zeros(self._desc_len, dtype=np.float64))
            else:
                des = np.asarray(des, dtype=np.float64)
                feats.append(des.mean(axis=0))
        return Features(X=np.stack(feats, axis=0), name=self.name)


class ORBExtractor(_MeanDescriptorExtractor):
    _kind = "orb"
    _desc_len = 32
    name = "ORB(cv2)"


class SIFTExtractor(_MeanDescriptorExtractor):
    _kind = "sift"
    _desc_len = 128
    name = "SIFT(cv2)"
