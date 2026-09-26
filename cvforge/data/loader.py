"""data/loader — 图像目录 / CSV 载入（DatasetSource 实现）。

可选依赖：Pillow（图像 IO）。缺失时 available()=False，benchmark 自动跳过；
合成数据路径不依赖本模块 → 零下载 demo 仍可跑。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cvforge.core.types import ImageDataset


def _have_pil() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except Exception:
        return False


def _load_gray(path: str, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L").resize((size, size))
        arr = np.asarray(im, dtype=np.float64) / 255.0
    return arr


class ImageFolderSource:
    """按子目录=类别载入图像数据集。"""

    def __init__(self, root: str, img_size: int = 32) -> None:
        self.root = root
        self.img_size = int(img_size)

    def available(self) -> bool:
        return _have_pil()

    def load(self) -> ImageDataset:
        import os

        if not self.available():
            raise RuntimeError("Pillow 不可用，无法载入图像目录")
        class_names: List[str] = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        images: list = []
        labels: list = []
        for idx, name in enumerate(class_names):
            d = os.path.join(self.root, name)
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    images.append(_load_gray(os.path.join(d, fn), self.img_size))
                    labels.append(idx)
        return ImageDataset(
            images=np.stack(images, axis=0),
            labels=np.asarray(labels, dtype=np.int64),
            class_names=class_names,
            name="image_folder",
        )


class CSVDataSource:
    """从 CSV 载入（img_col 路径 + label_col 类别名/索引）。"""

    def __init__(
        self,
        path: str,
        img_col: str = "path",
        label_col: str = "label",
        img_size: int = 32,
    ) -> None:
        self.path = path
        self.img_col = img_col
        self.label_col = label_col
        self.img_size = int(img_size)

    def available(self) -> bool:
        return _have_pil()

    def load(self) -> ImageDataset:
        import os

        import pandas as pd

        df = pd.read_csv(self.path)
        label_names: List[str] = sorted(df[self.label_col].astype(str).unique())
        name_to_idx = {n: i for i, n in enumerate(label_names)}
        images: list = []
        labels: list = []
        for _, row in df.iterrows():
            p = str(row[self.img_col])
            if not os.path.isabs(p) and not os.path.exists(p):
                p = os.path.join(os.path.dirname(self.path), p)
            images.append(_load_gray(p, self.img_size))
            labels.append(name_to_idx[str(row[self.label_col])])
        return ImageDataset(
            images=np.stack(images, axis=0),
            labels=np.asarray(labels, dtype=np.int64),
            class_names=label_names,
            name="csv",
        )


def load_image_folder(root: str, img_size: int = 32) -> ImageDataset:
    return ImageFolderSource(root, img_size).load()


def load_csv(path: str, **kwargs) -> ImageDataset:
    return CSVDataSource(path, **kwargs).load()
