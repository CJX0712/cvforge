"""tests/test_models — 分类器契约与行为。"""

import numpy as np

from cvforge.models.baselines import MajorityBaseline, RandomBaseline
from cvforge.models.sklearn_classifiers import make_knn, make_random_forest, make_svm


def _separable():
    # 两类明显可分
    X = np.vstack([np.random.default_rng(0).normal(0, 1, (50, 4)),
                   np.random.default_rng(1).normal(5, 1, (50, 4))])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def test_rf_svm_knn_fit_predict_separable():
    X, y = _separable()
    for fac in (make_random_forest(42), make_svm(42), make_knn()):
        clf = fac
        assert clf.available() is True
        clf.fit(X, y)
        pred = clf.predict(X)
        assert np.mean(pred.y_pred == y) > 0.95


def test_majority_baseline_constant():
    y = np.array([0, 0, 0, 1, 1, 2])
    m = MajorityBaseline()
    m.fit(np.zeros((6, 3)), y)
    p = m.predict(np.zeros((4, 3)))
    assert set(np.unique(p.y_pred).tolist()) == {0}
    assert len(p.y_pred) == 4


def test_random_baseline_in_range():
    y = np.array([0, 1, 2, 3])
    r = RandomBaseline(42)
    r.fit(np.zeros((4, 3)), y)
    p = r.predict(np.zeros((10, 3)))
    assert set(np.unique(p.y_pred).tolist()).issubset({0, 1, 2, 3})


def test_predict_before_fit_raises():
    from cvforge.models.base import BaseClassifier

    clf = make_random_forest(42)
    try:
        clf.predict(np.zeros((2, 4)))
    except Exception:
        return
    raise AssertionError("expected error on predict before fit")
