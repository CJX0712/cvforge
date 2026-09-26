"""eval/metrics — 分类与聚类指标（复用 scikit-learn，非自研）。

- accuracy / macro-F1 / balanced-accuracy：有监督分类（越高越好）。
- NMI / ARI / AMI：无监督聚类对齐真实标签（越高越好）。

公式与实现均来自 scikit-learn（BSD），符合「复用/不重复造轮子」原则。
导入使用别名以避免与本地函数名递归。
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import accuracy_score as _acc
from sklearn.metrics import adjusted_mutual_info_score as _ami
from sklearn.metrics import adjusted_rand_score as _ari
from sklearn.metrics import balanced_accuracy_score as _bal
from sklearn.metrics import f1_score as _f1
from sklearn.metrics import normalized_mutual_info_score as _nmi

METRICS: dict = {
    "ACC": "Accuracy",
    "F1": "Macro F1-score",
    "BAL_ACC": "Balanced Accuracy",
    "NMI": "Normalized Mutual Info",
    "ARI": "Adjusted Rand Index",
    "AMI": "Adjusted Mutual Info",
}


def accuracy(actual, pred) -> float:
    return float(_acc(actual, pred))


def macro_f1(actual, pred) -> float:
    return float(_f1(actual, pred, average="macro", zero_division=0))


def balanced_acc(actual, pred) -> float:
    return float(_bal(actual, pred))


def clustering_metrics(actual, pred) -> tuple:
    """返回 (NMI, ARI, AMI)。"""
    return (
        float(_nmi(actual, pred)),
        float(_ari(actual, pred)),
        float(_ami(actual, pred)),
    )


def evaluate_classification(actual, pred) -> dict:
    """有监督分类三指标字典。"""
    return {
        "ACC": accuracy(actual, pred),
        "F1": macro_f1(actual, pred),
        "BAL_ACC": balanced_acc(actual, pred),
    }
