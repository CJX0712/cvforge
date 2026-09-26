"""examples/run_demo — 最小可运行示例（clone 后零干预可跑）。

演示：合成数据 -> 多特征 -> 多分类器 -> 打印表格 -> 写 benchmark.json。
运行：python cvforge/examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

# 将仓库根加入 sys.path（无需安装即可运行）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cvforge.pipeline.cvforge_pipeline import CVForgePipeline  # noqa: E402


def main() -> None:
    pipe = CVForgePipeline()
    result, dataset = pipe.run()
    print(f"dataset: {dataset.name} ({dataset.n_samples} imgs, {dataset.n_classes} classes)")
    print(result.to_table(metrics_order=["ACC", "F1", "BAL_ACC"]))
    out = os.path.join(_REPO_ROOT, "benchmark.json")
    result.save_json(out)
    print(f"\nwritten: {out} ({len(result.rows)} models)")


if __name__ == "__main__":
    main()
