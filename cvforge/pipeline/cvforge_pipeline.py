"""pipeline/cvforge_pipeline — 端到端编排。

数据(合成) -> 多特征提取 -> 多分类器(含基线) -> 分层评估 -> 基准表/JSON；
并附 KMeans 无监督聚类对齐(NMI/ARI)作为第二维度。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from cvforge.core.config import Config
from cvforge.core.types import BenchmarkResult, ImageDataset
from cvforge.data.synthetic import make_synthetic
from cvforge.eval.benchmark import benchmark, evaluate_clustering
from cvforge.features.hog import HOGExtractor
from cvforge.features.keypoint import ORBExtractor, SIFTExtractor
from cvforge.features.pixel import PixelFlattenExtractor
from cvforge.models.baselines import MajorityBaseline, RandomBaseline
from cvforge.models.sklearn_classifiers import (
    make_knn,
    make_random_forest,
    make_svm,
)


class CVForgePipeline:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    # ---- 特征提取器（按配置 + available 探测） ----
    def default_extractors(self) -> List:
        cfg = self.config
        ex: List = []
        if cfg.use_pixel:
            ex.append(PixelFlattenExtractor())
        if cfg.use_hog:
            h = HOGExtractor()
            if h.available():
                ex.append(h)
        if cfg.use_orb:
            o = ORBExtractor()
            if o.available():
                ex.append(o)
        if cfg.use_sift:
            s = SIFTExtractor()
            if s.available():
                ex.append(s)
        return ex

    # ---- 分类器工厂（每组合新建实例，避免跨特征共享拟合态） ----
    def default_classifiers(self) -> List[Tuple[str, Callable[[], object]]]:
        cfg = self.config
        rs = cfg.random_state
        out: List[Tuple[str, Callable[[], object]]] = []
        if cfg.use_rf:
            out.append(("RandomForest", lambda: make_random_forest(rs)))
        if cfg.use_svm:
            out.append(("SVM(RBF)", lambda: make_svm(rs)))
        if cfg.use_knn:
            out.append(("KNN", lambda: make_knn()))
        out.append(("Majority", lambda: MajorityBaseline()))
        out.append(("Random", lambda: RandomBaseline(rs)))
        return out

    def run(
        self,
        dataset: Optional[ImageDataset] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[BenchmarkResult, ImageDataset]:
        if dataset is None:
            dataset = make_synthetic(self.config)
        y = dataset.labels
        extractors = self.default_extractors()
        classifiers = self.default_classifiers()

        all_rows: List[dict] = []
        cluster_feat = None
        for ext in extractors:
            feats = ext.extract(dataset)
            X = feats.X
            labeled = [
                (f"{ext.name}@{lbl}", factory())
                for lbl, factory in classifiers
            ]
            res = benchmark(labeled, X, y, self.config, verbose=True)
            all_rows.extend(res.rows)
            # 选聚类对齐的展示特征：优先 HOG（判别性最强），否则首个可用
            if cluster_feat is None:
                cluster_feat = ext
            if ext.name.startswith("HOG"):
                cluster_feat = ext

        # 无监督聚类对齐（KMeans vs 真实标签），作为第二评测维度
        if cluster_feat is not None:
            try:
                Xc = cluster_feat.extract(dataset).X
                cm = evaluate_clustering(
                    Xc, y, n_clusters=dataset.n_classes,
                    random_state=self.config.random_state,
                )
                cluster_summary = [
                    f"{cluster_feat.name}: NMI={cm['NMI']:.4f} ARI={cm['ARI']:.4f} AMI={cm['AMI']:.4f}"
                ]
            except Exception:
                cluster_summary = []

        if cluster_summary:
            print("\n[clustering] KMeans vs true labels:")
            for line in cluster_summary:
                print("  " + line)

        result = BenchmarkResult(rows=all_rows)
        if save_path:
            result.save_json(save_path)
        return result, dataset

    def run_demo(self, save_path: str = "benchmark.json") -> BenchmarkResult:
        res, _ = self.run(save_path=save_path)
        return res
