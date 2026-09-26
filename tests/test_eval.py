"""tests/test_eval — 分类/聚类指标正确性。"""

import numpy as np

from cvforge.core.config import Config
from cvforge.eval.benchmark import benchmark, evaluate_classifier
from cvforge.eval.metrics import accuracy, balanced_acc, clustering_metrics, macro_f1
from cvforge.models.baselines import MajorityBaseline
from cvforge.models.sklearn_classifiers import make_random_forest


def test_accuracy_perfect_and_worst():
    y = np.array([0, 1, 2, 0, 1])
    assert abs(accuracy(y, y) - 1.0) < 1e-12
    assert abs(accuracy(y, np.array([1, 0, 0, 1, 0])) - 0.0) < 1e-12


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([0, 1, 2, 1, 0, 2])
    assert abs(macro_f1(y, p) - f1_score(y, p, average="macro", zero_division=0)) < 1e-12


def test_clustering_metrics_perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    nmi, ari, ami = clustering_metrics(y, y)
    assert abs(nmi - 1.0) < 1e-9
    assert abs(ari - 1.0) < 1e-9
    assert abs(ami - 1.0) < 1e-9


def test_evaluate_classifier_returns_three_metrics():
    from cvforge.data.synthetic import make_synthetic

    ds = make_synthetic(Config())
    from cvforge.features.pixel import PixelFlattenExtractor

    X = PixelFlattenExtractor().extract(ds).X
    res = evaluate_classifier("Pixel@RF", make_random_forest(42), X, ds.labels, Config())
    names = {r.name for r in res}
    assert names == {"ACC", "F1", "BAL_ACC"}


def test_benchmark_skips_unavailable(monkeypatch):
    from cvforge.features.pixel import PixelFlattenExtractor
    from cvforge.models.sklearn_classifiers import make_random_forest

    # 让 RandomForest 不可用 -> 应被跳过
    monkeypatch.setattr(
        type(make_random_forest(42)), "available", lambda self: False
    )
    X = np.random.default_rng(0).random((20, 10))
    y = np.array([0, 1] * 10)
    labeled = [("RF", make_random_forest(42))]
    res = benchmark(labeled, X, y, Config(), verbose=False)
    assert res.rows == []


def test_hog_rf_high_accuracy():
    """SOTA 特征(HOG) + RF 在合成形状上应远高于随机基线。"""
    from cvforge.data.synthetic import make_synthetic
    from cvforge.features.hog import HOGExtractor

    ds = make_synthetic(Config())
    X = HOGExtractor().extract(ds).X
    res = evaluate_classifier("HOG@RF", make_random_forest(42), X, ds.labels, Config())
    acc = {r.name: r.value for r in res}["ACC"]
    assert acc > 0.85
