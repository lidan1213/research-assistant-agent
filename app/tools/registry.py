"""工具注册表聚合。

导入本模块会触发所有内建工具的注册，并返回全局 `registry` 实例。
"""
from app.tools.base import registry

# 触碰各工具模块以完成注册（顺序无关）
from app.tools import (  # noqa: E402,F401
    calculator,
    citation,
    code_executor,
    knowledge_search,
    pdf_reader,
    web_search,
    arxiv_search,
    kg_search,
)

__all__ = ["registry"]
