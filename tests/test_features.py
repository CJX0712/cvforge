"""tests/test_features — 特征提取器契约与输出形状。"""

import numpy as np
import pytest

from cvforge.data.synthetic import make_synthetic
from cvforge.features.hog import HOGExtractor
from cvforge.features.keypoint import ORBExtractor, SIFTExtractor
from cvforge.features.pixel import PixelFlattenExtractor


@pytest.fixture
def ds():
    return make_synthetic()


def test_pixel_always_available_and_shape(ds):
    ext = PixelFlattenExtractor()
    assert ext.available() is True
    f = ext.extract(ds)
    assert f.X.shape == (ds.n_samples, 32 * 32)


def test_hog_available_and_shape(ds):
    ext = HOGExtractor()
    assert ext.available() is True
    f = ext.extract(ds)
    # (32/8 - 1)^2 * 9 * (2*2) = 3*3*9*4 = 324
    assert f.X.shape == (ds.n_samples, 324)


def test_orb_available_and_shape(ds):
    ext = ORBExtractor()
    assert ext.available() is True
    f = ext.extract(ds)
    assert f.X.shape == (ds.n_samples, 32)
    # 修正前默认 ORB 在小图上漏掉全部关键点 -> 全零；现应非零行占多数
    nonzero = int((np.abs(f.X).sum(axis=1) > 0).sum())
    assert nonzero > 0.8 * ds.n_samples


def test_sift_available_and_shape(ds):
    ext = SIFTExtractor()
    assert ext.available() is True
    f = ext.extract(ds)
    assert f.X.shape == (ds.n_samples, 128)
