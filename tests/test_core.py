"""tests/test_core — 核心类型、配置、错误码契约。"""

import json
import os

import numpy as np
import pytest

from cvforge.core.config import Config, ENV_PREFIX
from cvforge.core.errors import (
    ConfigError,
    DataError,
    EvalError,
    ModelError,
    CVForgeError,
)
from cvforge.core.types import (
    BenchmarkResult,
    Features,
    ImageDataset,
    Prediction,
)


# ---------- ImageDataset ----------
def test_imagedataset_shapes_and_labels():
    imgs = np.random.default_rng(0).random((10, 16, 16))
    ds = ImageDataset(images=imgs, labels=np.arange(10) % 3, class_names=["a", "b", "c"])
    assert ds.n_samples == 10
    assert ds.n_classes == 3
    assert ds.img_shape == (16, 16)


def test_imagedataset_4d_collapses_to_3d():
    imgs = np.random.default_rng(0).random((5, 8, 8, 1))
    ds = ImageDataset(images=imgs, labels=np.zeros(5, dtype=int), class_names=["x"])
    assert ds.images.ndim == 3
    assert ds.img_shape == (8, 8)


# ---------- Features / Prediction ----------
def test_features_matrix():
    f = Features(X=np.zeros((3, 5)), name="t")
    assert f.X.shape == (3, 5)


def test_prediction_y_pred_1d():
    p = Prediction(y_pred=[0, 1, 2])
    assert p.y_pred.shape == (3,)
    assert p.y_proba is None


# ---------- BenchmarkResult ----------
def test_benchmark_to_table_contains_models():
    br = BenchmarkResult(
        rows=[
            {"model": "PixelFlatten@RandomForest", "ACC": 0.68, "F1": 0.67},
            {"model": "HOG(skimage)@RandomForest", "ACC": 0.98, "F1": 0.98},
        ]
    )
    table = br.to_table()
    assert "PixelFlatten@RandomForest" in table
    assert "HOG(skimage)@RandomForest" in table
    assert "ACC" in table


def test_benchmark_to_table_empty():
    assert BenchmarkResult(rows=[]).to_table() == "(no results)"


def test_benchmark_save_json_roundtrip(tmp_path):
    br = BenchmarkResult(rows=[{"model": "X", "ACC": 0.9}])
    p = tmp_path / "b.json"
    br.save_json(str(p))
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["rows"][0]["model"] == "X"


# ---------- Config ----------
def test_config_defaults():
    c = Config()
    assert c.n_per_class == 40
    assert c.img_size == 32
    assert c.classes == ("circle", "triangle", "square", "cross")
    assert c.test_size == 0.30


def test_config_from_env_override(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}N_PER_CLASS", "60")
    monkeypatch.setenv(f"{ENV_PREFIX}TEST_SIZE", "0.2")
    monkeypatch.setenv(f"{ENV_PREFIX}CLASSES", "circle,square")
    c = Config.from_env()
    assert c.n_per_class == 60
    assert abs(c.test_size - 0.2) < 1e-12
    assert c.classes == ("circle", "square")


# ---------- Errors ----------
def test_error_hierarchy_and_codes():
    assert issubclass(ConfigError, CVForgeError)
    assert issubclass(DataError, CVForgeError)
    assert issubclass(ModelError, CVForgeError)
    assert issubclass(EvalError, CVForgeError)
    assert CVForgeError().code == "E000"
    assert ConfigError().code == "E100"
    assert DataError().code == "E200"
    assert ModelError().code == "E300"
    assert EvalError().code == "E400"
