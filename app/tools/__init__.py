"""工具系统。

包含：工具基类、自动 JSON Schema 生成、注册表，以及一组合科研向工具。
框架会自动把注册的工具转换为 LLM 的 function-calling schema。
"""
from app.tools.base import (
    BaseTool,
    FunctionTool,
    ToolRegistry,
    ToolResult,
    tool,
)
from app.tools.registry import registry

# 导入所有内建工具以触发注册
from app.tools import web_search, arxiv_search, pdf_reader, code_executor, calculator, citation, kg_search  # noqa: F401

__all__ = [
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "ToolResult",
    "tool",
    "registry",
]
