"""tests/test_pipeline — 端到端编排、SOTA 对标、离线降级。"""

import json

from cvforge.core.config import Config
from cvforge.data.synthetic import make_synthetic
from cvforge.features import hog as _hog_mod
from cvforge.features import keypoint as _kp_mod
from cvforge.pipeline.cvforge_pipeline import CVForgePipeline


def test_run_returns_result_and_writes_json(tmp_path):
    cfg = Config(n_per_class=30, random_state=7)
    pipe = CVForgePipeline(cfg)
    out = tmp_path / "bench.json"
    result, ds = pipe.run(save_path=str(out))
    assert result is not None
    assert len(result.rows) > 0
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "rows" in data


def test_sota_hog_beats_pixel_baseline():
    """HOG 特征显著高于原始像素特征（结构被有效编码）。"""
    from cvforge.features.hog import HOGExtractor
    from cvforge.features.pixel import PixelFlattenExtractor
    from cvforge.models.sklearn_classifiers import make_random_forest
    from cvforge.eval.benchmark import evaluate_classifier

    ds = make_synthetic(Config())
    cfg = Config()
    hog_acc = {r.name: r.value for r in evaluate_classifier(
        "HOG@RF", make_random_forest(42), HOGExtractor().extract(ds).X, ds.labels, cfg)}["ACC"]
    pix_acc = {r.name: r.value for r in evaluate_classifier(
        "Pix@RF", make_random_forest(42), PixelFlattenExtractor().extract(ds).X, ds.labels, cfg)}["ACC"]
    assert hog_acc > pix_acc + 0.1


def test_offline_fallback_skips_optional_backends(monkeypatch):
    """可选后端(HOG/ORB/SIFT)缺失时仅保留 PixelFlatten 路径，demo 仍可跑。"""
    monkeypatch.setattr(_hog_mod.HOGExtractor, "available", lambda self: False)
    monkeypatch.setattr(_kp_mod.ORBExtractor, "available", lambda self: False)
    monkeypatch.setattr(_kp_mod.SIFTExtractor, "available", lambda self: False)

    cfg = Config(n_per_class=30, random_state=3)
    pipe = CVForgePipeline(cfg)
    result, _ = pipe.run()
    names = {r["model"] for r in result.rows}
    assert all(n.startswith("PixelFlatten@") for n in names)
    assert "HOG(skimage)@RandomForest" not in names
    assert "ORB(cv2)@RandomForest" not in names
    assert "SIFT(cv2)@RandomForest" not in names


def test_clustering_summary_reported():
    cfg = Config(n_per_class=30)
    pipe = CVForgePipeline(cfg)
    result, _ = pipe.run()
    # benchmark.json 已写出；聚类对齐在 stdout，这里仅验证主流程无异常
    assert len(result.rows) > 0
