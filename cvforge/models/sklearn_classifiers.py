"""models/sklearn_classifiers — 复用 scikit-learn 分类器（SOTA 监督分支）。

复用 RandomForest / SVM / KNN 等工业级分类器（BSD-3，已为依赖，零额外体积、稳定），
属「集成领先开源」而非自研模型。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cvforge.core.types import Prediction
from cvforge.models.base import BaseClassifier


class SklearnClassifier(BaseClassifier):
    def __init__(self, estimator, name: Optional[str] = None) -> None:
        super().__init__()
        self._est = estimator
        self.name = name or type(estimator).__name__

    def available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, X, y) -> "SklearnClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        self._est.fit(X, y)
        self._classes = np.unique(y)
        self._fitted = True
        return self

    def predict(self, X) -> Prediction:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float64)
        yp = np.asarray(self._est.predict(X), dtype=np.int64).ravel()
        proba = None
        if hasattr(self._est, "predict_proba"):
            try:
                proba = np.asarray(self._est.predict_proba(X), dtype=np.float64)
            except Exception:
                proba = None
        return Prediction(y_pred=yp, y_proba=proba)


def make_random_forest(random_state: int = 42, n_estimators: int = 200):
    from sklearn.ensemble import RandomForestClassifier

    return SklearnClassifier(
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        name="RandomForest",
    )


def make_svm(random_state: int = 42, C: float = 10.0):
    from sklearn.svm import SVC

    return SklearnClassifier(
        SVC(C=C, kernel="rbf", random_state=random_state),
        name="SVM(RBF)",
    )


def make_knn(n_neighbors: int = 5):
    from sklearn.neighbors import KNeighborsClassifier

    return SklearnClassifier(
        KNeighborsClassifier(n_neighbors=n_neighbors), name="KNN"
    )
