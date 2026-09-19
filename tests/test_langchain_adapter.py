"""LangChain 桥接适配器测试：消息互转 / 工具包装 / provider 接入。"""
from __future__ import annotations

import json

from app.llm.base import ChatMessage, MessageRole, ToolCall
from app.llm.langchain_adapter import from_langchain, to_langchain


def _roundtrip(m: ChatMessage) -> ChatMessage:
    return from_langchain(to_langchain(m))


def test_message_roundtrip_user():
    m = _roundtrip(ChatMessage(role=MessageRole.USER, content="量子点是什么"))
    assert m.role == MessageRole.USER
    assert m.content == "量子点是什么"


def test_message_roundtrip_system_and_tool():
    s = _roundtrip(ChatMessage(role=MessageRole.SYSTEM, content="你是科研助手"))
    assert s.role == MessageRole.SYSTEM and s.content == "你是科研助手"
    t = _roundtrip(ChatMessage(role=MessageRole.TOOL, content="391", tool_call_id="call_1"))
    assert t.role == MessageRole.TOOL and t.content == "391" and t.tool_call_id == "call_1"


def test_message_roundtrip_assistant_tool_calls():
    m = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="call_x", name="calculator", arguments='{"expr": "23*17"}')],
    )
    r = _roundtrip(m)
    assert r.role == MessageRole.ASSISTANT
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].id == "call_x"
    assert r.tool_calls[0].name == "calculator"
    assert json.loads(r.tool_calls[0].arguments) == {"expr": "23*17"}


def test_message_roundtrip_multimodal_user():
    """多模态 user（图片数组）→ langchain → 回来时图片丢失但文本保留（langchain 无图片字段通道）。"""
    m = ChatMessage(
        role=MessageRole.USER,
        content="图里有什么",
        images=["data:image/png;base64,abc"],
    )
    lc = to_langchain(m)
    # langchain HumanMessage 的多模态 content 是 list
    assert isinstance(lc.content, list)
    assert lc.content[0]["type"] == "text"
    assert lc.content[1]["type"] == "image_url"
    r = from_langchain(lc)
    assert r.content == "图里有什么"
    assert r.role == MessageRole.USER


def test_langchain_tool_wrapper():
    """LangChain 工具（StructuredTool）→ 自研 BaseTool 包装。"""
    from langchain_core.tools import StructuredTool

    async def _calc(expr: str) -> str:
        return f"calc({expr})=42"

    lc_tool = StructuredTool.from_function(
        coroutine=_calc,
        name="lc_calc",
        description="计算表达式",
        args_schema=None,
    )
    from app.tools.langchain_adapter import LangChainToolWrapper

    import asyncio

    wt = LangChainToolWrapper(lc_tool)
    assert wt.name == "lc_calc"
    assert "计算表达式" in wt.description
    schema = wt.parameters_schema()
    assert schema.get("type") == "object"
    out = asyncio.run(wt.run(expr="1+1"))
    assert "42" in str(out)


def test_get_llm_langchain_provider():
    """get_llm(provider='langchain') 返回 LangChain 桥实现（不发起真实调用）。"""
    from app.llm.factory import get_llm

    llm = get_llm(provider="langchain", model="gpt-4o-mini")
    from app.llm.langchain_adapter import LangChainChatModel

    assert isinstance(llm, LangChainChatModel)
