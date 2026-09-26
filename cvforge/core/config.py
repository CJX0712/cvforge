"""config — 全局配置（支持环境变量覆盖 CVFORGE_*）。

所有可调超参集中于此，模块只读 Config，不自行散落常量，便于复现与基准控制。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "CVFORGE_"


@dataclass
class Config:
    # 合成数据
    random_state: int = 42
    n_per_class: int = 40           # 每类样本数
    img_size: int = 32              # 图像边长（灰度方阵）
    classes: tuple = ("circle", "triangle", "square", "cross")

    # 评测
    test_size: float = 0.30         # 分层测试集比例

    # 特征（运行时按 available() 探测）
    use_pixel: bool = True
    use_hog: bool = True
    use_orb: bool = True
    use_sift: bool = True

    # 分类器
    use_rf: bool = True
    use_svm: bool = True
    use_knn: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """用环境变量覆盖：CVFORGE_N_PER_CLASS=60 等。"""
        overrides: dict = {}
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        for name in fields:
            env = ENV_PREFIX + name.upper()
            if env in os.environ:
                overrides[name] = _coerce(os.environ[env], fields[name].type)
        return cls(**overrides)


def _coerce(raw: str, typ):
    raw = raw.strip()
    t = str(typ)
    if "int" in t:
        return int(raw)
    if "float" in t:
        return float(raw)
    if "bool" in t:
        return raw.lower() in ("1", "true", "yes", "y")
    if "tuple" in t or "list" in t:
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return raw
