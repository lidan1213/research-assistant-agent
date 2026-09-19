"""把本项目注册表中的工具转换为 LangChain 工具，并附带执行调度。"""
from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from app.tools.registry import registry as default_registry

# JSON Schema 基础类型 -> Python 类型
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_model(name: str, schema: dict):
    """将工具的 JSON Schema 参数转换为 Pydantic 模型（供 LangChain 校验）。"""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for pname, pdef in props.items():
        pytype = _TYPE_MAP.get(pdef.get("type"), Any)
        if pname in required:
            fields[pname] = (pytype, ...)
        else:
            fields[pname] = (pytype | None, None)
    return create_model(f"{name}Args", **fields)


def build_langchain_tools(reg=None) -> list[StructuredTool]:
    """从工具注册表构建 LangChain 工具列表（自动生成参数 schema）。

    执行统一走 ToolExecutor：LangGraph 侧的 StructuredTool 与自研 ReAct
    共用同一套护栏（超时/重试/连续失败/重复调用/截断/追踪）。
    """
    reg = reg or default_registry
    from app.tools.executor import ToolExecutor

    executor = ToolExecutor(
        reg,
        timeout=90,
        max_retries=1,
        max_chars=8000,
    )
    out: list[StructuredTool] = []
    for t in reg.list():
        model = _schema_to_model(t.name, t.parameters_schema())

        async def _run(name=t.name, **kwargs):
            return await executor.execute(name, json.dumps(kwargs, ensure_ascii=False))

        out.append(
            StructuredTool.from_function(
                name=t.name,
                description=t.description,
                coroutine=_run,
                args_schema=model,
            )
        )
    return out


async def dispatch_tool(tool_call: dict, tools: list[StructuredTool]) -> ToolMessage:
    """执行单个 LangChain tool_call，返回可被追加到消息历史的 ToolMessage。

    使用 `ainvoke` 以支持异步工具（本项目工具底层为协程），可在异步图节点中直接 await。
    """
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        content = f"未找到工具: {name}"
    else:
        try:
            content = await tool.ainvoke(args)
        except Exception as e:  # noqa: BLE001
            content = f"工具执行错误: {e}"
    return ToolMessage(
        content=str(content),
        tool_call_id=tool_call.get("id", ""),
        name=name,
    )


def make_handoff(member: str) -> StructuredTool:
    """构造一个「把控制权交给某成员」的 LangChain 工具，用于主管路由。

    支持可选 `payload` 参数：主管可在委派时显式携带一段需要直传给该专家的中间产物/指令，
    与共享黑板 + inbox 自动直传互补（前者自动广播上游成果，后者可定向携带特定内容）。
    """
    args = create_model(
        f"{member}HandoffArgs",
        payload=(Optional[str], Field(None, description="可选：随委派直传给该专家的中间产物/指令")),
    )

    async def _handoff(payload: Optional[str] = None) -> str:
        return f"已转移给 {member}"

    return StructuredTool.from_function(
        name=f"transfer_to_{member}",
        description=f"将任务移交给 {member} 专家处理；可通过 payload 携带需要直传的中间产物或指令。",
        coroutine=_handoff,
        args_schema=args,
    )
