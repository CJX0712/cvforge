"""tests/test_data — 合成数据生成确定性、形状、标签正确性。"""

import numpy as np

from cvforge.core.config import Config
from cvforge.data.synthetic import make_synthetic


def test_synthetic_shape_and_labels():
    ds = make_synthetic(Config())
    assert ds.images.ndim == 3
    assert ds.images.shape[0] == 4 * 40
    assert ds.images.shape[1:] == (32, 32)
    assert ds.n_classes == 4
    assert set(np.unique(ds.labels).tolist()) == {0, 1, 2, 3}
    # 每类样本数相等
    _, counts = np.unique(ds.labels, return_counts=True)
    assert counts.min() == counts.max() == 40


def test_synthetic_values_in_unit_range():
    ds = make_synthetic(Config())
    assert ds.images.min() >= 0.0
    assert ds.images.max() <= 1.0


def test_synthetic_deterministic():
    a = make_synthetic(Config())
    b = make_synthetic(Config())
    assert np.allclose(a.images, b.images)
    assert np.array_equal(a.labels, b.labels)


def test_synthetic_custom_classes():
    ds = make_synthetic(Config(), classes=("circle", "cross"), n_per_class=10)
    assert ds.n_classes == 2
    assert ds.n_samples == 20
