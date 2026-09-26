"""errors — 分层错误码（E1xx 体系，与 recforge/tsforge 同构）。

使用：抛出具体子类；上层按 code 区分处理。
"""

from __future__ import annotations


class CVForgeError(Exception):
    """基类。"""

    code = "E000"


class ConfigError(CVForgeError):
    code = "E100"


class DataError(CVForgeError):
    code = "E200"


class ModelError(CVForgeError):
    code = "E300"


class EvalError(CVForgeError):
    code = "E400"
