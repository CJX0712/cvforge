"""eval/benchmark — 跨「特征@分类器」基准编排。

遍历 (标签, 分类器) → 跳过不可用/失败 → 分层切分 → 训练 → 预测 → 聚合为 BenchmarkResult。
聚类对齐（NMI/ARI/AMI）单独评估 KMeans，作为无监督维度补充。
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from cvforge.core.config import Config
from cvforge.core.types import BenchmarkResult, MetricResult
from cvforge.eval.metrics import accuracy, balanced_acc, clustering_metrics, macro_f1


def evaluate_classifier(
    label: str,
    clf,
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
) -> List[MetricResult]:
    """对单个分类器在特征 X / 标签 y 上做分层评估，返回指标列表。"""
    from sklearn.model_selection import train_test_split

    H = X.shape[0]
    if H < 4:
        return []
    strat = y if len(np.unique(y)) > 1 else None
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=config.test_size,
            stratify=strat,
            random_state=config.random_state,
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        yp = np.asarray(pred.y_pred, dtype=np.int64).ravel()
        return [
            MetricResult("ACC", accuracy(yte, yp), label),
            MetricResult("F1", macro_f1(yte, yp), label),
            MetricResult("BAL_ACC", balanced_acc(yte, yp), label),
        ]
    except Exception:
        return []


def evaluate_clustering(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> dict:
    """KMeans 聚类对齐真实标签（无监督维度）。"""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=10)
    pred = km.fit_predict(X)
    nmi, ari, ami = clustering_metrics(y, pred)
    return {"NMI": nmi, "ARI": ari, "AMI": ami}


def benchmark(
    labeled_models: Iterable[Tuple[str, object]],
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
    verbose: bool = True,
) -> BenchmarkResult:
    rows_list: List[dict] = []
    for label, model in labeled_models:
        available = getattr(model, "available", lambda: True)()
        if not available:
            if verbose:
                print(f"[skip] {label} (optional backend missing)")
            continue
        try:
            res = evaluate_classifier(label, model, X, y, config)
        except Exception as e:  # 单模型失败不拖累其它
            if verbose:
                print(f"[error] {label} eval failed: {e}")
            continue
        if not res:
            if verbose:
                print(f"[warn] {label} produced no results")
            continue
        agg = {"model": label}
        for r in res:
            agg[r.name] = round(float(r.value), 4)
        rows_list.append(agg)
    return BenchmarkResult(rows=rows_list)
