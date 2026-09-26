"""data/synthetic — 合成形状图像（纯 numpy，无外部依赖，保证离线可生成）。

生成 4 类可区分几何形状（circle / triangle / square / cross），带位置/尺度/亮度抖动
与高斯噪声。因结构真实存在，HOG + 分类器应显著高于随机基线（acc≥0.95）。
固定 random_state 保证基准可复现。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cvforge.core.config import Config
from cvforge.core.types import ImageDataset


def _mesh(size: int):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return xx.astype(np.float64), yy.astype(np.float64)


def _draw_circle(xx, yy, rng, size):
    cx, cy = rng.uniform(10, size - 10), rng.uniform(10, size - 10)
    r = rng.uniform(6.0, 9.0)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _draw_square(xx, yy, rng, size):
    cx, cy = rng.uniform(10, size - 10), rng.uniform(10, size - 10)
    s = rng.uniform(6.0, 9.0)
    return (np.abs(xx - cx) <= s) & (np.abs(yy - cy) <= s)


def _draw_triangle(xx, yy, rng, size):
    h = rng.uniform(13.0, 18.0)
    w = rng.uniform(8.0, 11.0)
    cx = rng.uniform(w, size - w)
    cy = rng.uniform(h / 2, size - h / 2)
    top = cy - h / 2
    inside_y = (yy >= top) & (yy <= top + h)
    t = (yy - top) / h
    hw = (w / 2) * np.clip(t, 0.0, 1.0)
    return inside_y & (np.abs(xx - cx) <= hw)


def _draw_cross(xx, yy, rng, size):
    cx, cy = rng.uniform(12, size - 12), rng.uniform(12, size - 12)
    a = rng.uniform(3.0, 4.5)
    L = rng.uniform(9.0, 12.0)
    vert = (np.abs(xx - cx) <= a) & (np.abs(yy - cy) <= L)
    horiz = (np.abs(yy - cy) <= a) & (np.abs(xx - cx) <= L)
    return vert | horiz


_DRAW = {
    "circle": _draw_circle,
    "triangle": _draw_triangle,
    "square": _draw_square,
    "cross": _draw_cross,
}


def make_synthetic(
    config: Optional[Config] = None,
    *,
    n_per_class: Optional[int] = None,
    img_size: Optional[int] = None,
    classes: Optional[Tuple[str, ...]] = None,
    random_state: Optional[int] = None,
) -> ImageDataset:
    cfg = config or Config()
    n_per_class = int(n_per_class if n_per_class is not None else cfg.n_per_class)
    size = int(img_size if img_size is not None else cfg.img_size)
    classes = tuple(classes if classes is not None else cfg.classes)
    rs = int(random_state if random_state is not None else cfg.random_state)

    rng = np.random.default_rng(rs)
    xx, yy = _mesh(size)

    images: list = []
    labels: list = []
    for cls_idx, cls_name in enumerate(classes):
        if cls_name not in _DRAW:
            raise ValueError(f"unknown synthetic class '{cls_name}'")
        draw = _DRAW[cls_name]
        for _ in range(n_per_class):
            canvas = np.zeros((size, size), dtype=np.float64)
            mask = draw(xx, yy, rng, size)
            intensity = float(rng.uniform(0.6, 1.0))
            canvas[mask] = intensity
            canvas += rng.normal(0.0, 0.05, size=(size, size))
            canvas = np.clip(canvas, 0.0, 1.0)
            images.append(canvas)
            labels.append(cls_idx)

    return ImageDataset(
        images=np.stack(images, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=list(classes),
        name="synthetic_shapes",
    )
