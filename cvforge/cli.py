"""cli — 命令行入口（argparse）。

用法：
  python -m cvforge.cli demo
  python -m cvforge.cli evaluate --feature hog --model rf
  python -m cvforge.cli version
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from cvforge.core.config import Config
from cvforge.data.synthetic import make_synthetic
from cvforge.eval.benchmark import benchmark
from cvforge.features.hog import HOGExtractor
from cvforge.features.keypoint import ORBExtractor, SIFTExtractor
from cvforge.features.pixel import PixelFlattenExtractor
from cvforge.models.baselines import MajorityBaseline, RandomBaseline
from cvforge.models.sklearn_classifiers import make_knn, make_random_forest, make_svm
from cvforge.pipeline.cvforge_pipeline import CVForgePipeline

_FEATURES = {
    "pixel": PixelFlattenExtractor,
    "hog": HOGExtractor,
    "orb": ORBExtractor,
    "sift": SIFTExtractor,
}

_MODELS = {
    "rf": lambda rs: make_random_forest(rs),
    "svm": lambda rs: make_svm(rs),
    "knn": lambda rs: make_knn(),
    "majority": lambda rs: MajorityBaseline(),
    "random": lambda rs: RandomBaseline(rs),
}


def _resolve_feature(name: str):
    key = name.strip().lower()
    if key not in _FEATURES:
        return None
    inst = _FEATURES[key]()
    if not inst.available():
        return "unavailable"
    return inst


def _resolve_model(name: str, rs: int):
    key = name.strip().lower()
    if key not in _MODELS:
        return None
    inst = _MODELS[key](rs)
    if not inst.available():
        return "unavailable"
    return inst


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="cvforge", description="cvforge: 基于特征的图像分类系统（作者: 晨星）"
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="运行端到端基准 -> benchmark.json")
    sub.add_parser("version", help="显示版本")
    ev = sub.add_parser("evaluate", help="对指定特征+分类器生成评测")
    ev.add_argument("--feature", type=str, default="hog", help="pixel/hog/orb/sift")
    ev.add_argument("--model", type=str, default="rf", help="rf/svm/knn/majority/random")

    args = p.parse_args(argv)
    cfg = Config.from_env()

    if args.cmd == "version" or args.cmd is None:
        from cvforge import __version__

        print(f"cvforge {__version__}")
        return 0

    if args.cmd == "demo":
        pipe = CVForgePipeline(cfg)
        result, _ = pipe.run(save_path="benchmark.json")
        print(result.to_table(metrics_order=["ACC", "F1", "BAL_ACC"]))
        print(f"\nbenchmark.json written ({len(result.rows)} models)")
        return 0

    if args.cmd == "evaluate":
        feat = _resolve_feature(args.feature)
        if feat is None:
            print(f"[error] unknown feature '{args.feature}'. available: {', '.join(_FEATURES)}")
            return 2
        if feat == "unavailable":
            print(f"[error] feature '{args.feature}' backend unavailable")
            return 2
        model = _resolve_model(args.model, cfg.random_state)
        if model is None:
            print(f"[error] unknown model '{args.model}'. available: {', '.join(_MODELS)}")
            return 2
        if model == "unavailable":
            print(f"[error] model '{args.model}' backend unavailable")
            return 2
        ds = make_synthetic(cfg)
        X = feat.extract(ds).X
        res = benchmark([(f"{feat.name}@{model.name}", model)], X, ds.labels, cfg, verbose=False)
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
