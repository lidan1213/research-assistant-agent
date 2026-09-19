"""LangChain 工具桥：把任意 LangChain BaseTool（含 langchain_community 生态工具）
包装成本项目自研 BaseTool，注册进自研 ToolRegistry —— LangChain 生态 100+ 工具即插即用。

用法：
    from langchain_community.tools import WikipediaQueryRun
    from app.tools.langchain_adapter import LangChainToolWrapper
    from app.tools.base import registry

    registry.register(LangChainToolWrapper(WikipediaQueryRun(...)))
"""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class LangChainToolWrapper(BaseTool):
    """把 langchain BaseTool / RunnableTool 适配为自研 BaseTool。"""

    def __init__(self, lc_tool: Any, *, name: str | None = None, description: str | None = None):
        self._lc = lc_tool
        self.name = name or getattr(lc_tool, "name", "lc_tool")
        self.description = (
            description or getattr(lc_tool, "description", "") or getattr(lc_tool, "name", self.name)
        )

    def parameters_schema(self) -> dict:
        args = getattr(self._lc, "args", None)
        if args is not None:
            # langchain 工具 args 是 Pydantic 模型的 JSON Schema（dict）；
            # 有的工具只给 properties 展开（缺 type: object 包装），补全
            schema = args if isinstance(args, dict) else getattr(args, "model_json_schema", lambda: {})()
            if isinstance(schema, dict) and schema.get("type") != "object":
                schema = {"type": "object", "properties": schema, "required": list(schema.keys())}
            return schema or {"type": "object", "properties": {}}
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs) -> Any:
        try:
            fn = getattr(self._lc, "ainvoke", None)
            if fn:
                out = await fn(kwargs)
            else:  # 同步工具
                out = self._lc.invoke(kwargs)  # type: ignore[attr-defined]
            return str(out)
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"LangChain 工具调用失败: {e}", retryable=False)
