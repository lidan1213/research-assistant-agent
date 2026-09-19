"""LangChain 桥接适配器：把 LangChain 模型（langchain-openai ChatOpenAI）适配成本项目自研 BaseLLM 接口。

价值：
- 自研 Agent 主链路（app/agent）可以插拔使用 LangChain 模型，获得 langchain 生态能力
  （模型包装、重试、fallback、缓存、其他 LangChain 模型供应商）；
- 与 LangGraph 引擎（app/graph/llm_bridge）共用同一套模型配置，两套引擎模型层统一；
- 消息格式互转（ChatMessage <-> langchain BaseMessage）独立提供，供任何桥接场景复用。

用法：
    from app.llm.factory import get_llm
    llm = get_llm(provider="langchain")          # 显式指定 LangChain 模型
    llm = get_llm(provider="langchain", model="gpt-5.4-mini")
    或 .env 设 LLM__PROVIDER=langchain 全局切换。
"""
from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.llm.base import BaseLLM, ChatMessage, LLMResponse, MessageRole, StreamChunk, ToolCall
from app.llm.openai_provider import LLMError, _record_llm_trace

# ---------- 消息互转 ----------


def to_langchain(m: ChatMessage) -> Any:
    """自研 ChatMessage -> langchain BaseMessage。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if m.role == MessageRole.SYSTEM:
        return SystemMessage(content=m.content)
    if m.role == MessageRole.USER:
        if m.images:
            parts: list[dict] = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            for img in m.images:
                parts.append({"type": "image_url", "image_url": {"url": img}})
            return HumanMessage(content=parts)
        return HumanMessage(content=m.content)
    if m.role == MessageRole.ASSISTANT:
        if m.tool_calls:
            return AIMessage(
                content=m.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "args": json.loads(tc.arguments or "{}"),
                    }
                    for tc in m.tool_calls
                ],
            )
        return AIMessage(content=m.content)
    # TOOL
    return ToolMessage(content=m.content, tool_call_id=m.tool_call_id or "")


def from_langchain(msg: Any) -> ChatMessage:
    """langchain BaseMessage -> 自研 ChatMessage。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if isinstance(msg, AIMessage):
        tcs: list[ToolCall] | None = None
        raw_tcs = getattr(msg, "tool_calls", None) or []
        if raw_tcs:
            tcs = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=json.dumps(tc.get("args", {}), ensure_ascii=False),
                )
                for tc in raw_tcs
            ]
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=msg.content or "",
            tool_calls=tcs,
        )
    if isinstance(msg, ToolMessage):
        return ChatMessage(
            role=MessageRole.TOOL,
            content=msg.content or "",
            tool_call_id=getattr(msg, "tool_call_id", None),
        )
    if isinstance(msg, SystemMessage):
        return ChatMessage(role=MessageRole.SYSTEM, content=msg.content or "")
    if isinstance(msg, HumanMessage):
        content = msg.content
        if isinstance(content, list):  # 多模态数组 → 提取文本
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "".join(texts)
        return ChatMessage(role=MessageRole.USER, content=str(content))
    return ChatMessage(role=MessageRole.USER, content=str(getattr(msg, "content", "")))


def _wrap_tools(tools: list[dict] | None) -> list[dict] | None:
    """自研 tools（裸 function dict）-> OpenAI 完整格式（bind_tools 要求）。"""
    if not tools:
        return None
    return [{"type": "function", "function": t} for t in tools]


class LangChainChatModel(BaseLLM):
    """自研 BaseLLM 接口的 LangChain 实现（内部用 langchain_openai.ChatOpenAI）。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        from langchain_openai import ChatOpenAI

        s = get_settings().llm
        self.model = model or s.model
        self._llm = ChatOpenAI(
            model=self.model,
            base_url=base_url or s.base_url,
            api_key=api_key or s.api_key or "EMPTY",
            temperature=s.temperature if temperature is None else temperature,
            max_tokens=s.max_tokens if max_tokens is None else max_tokens,
            timeout=s.timeout if timeout is None else timeout,
            max_retries=2,
        )

    def _bind(self, tools: list[dict] | None, json_mode: bool):
        llm = self._llm
        wrapped = _wrap_tools(tools)
        if wrapped:
            llm = llm.bind_tools(wrapped, tool_choice="auto")
        if json_mode:
            llm = llm.bind(response_format={"type": "json_object"})
        return llm

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ):
        """纯文本流（BaseLLM 抽象要求）：逐 token 产出字符串，不解析工具调用。"""
        async for chunk in self._llm.astream([to_langchain(m) for m in messages]):
            if chunk.content:
                yield chunk.content

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        import time

        _t0 = time.monotonic()
        try:
            llm = self._bind(tools, json_mode)
            resp = await llm.ainvoke([to_langchain(m) for m in messages])
        except Exception as e:  # noqa: BLE001
            # 部分供应商不支持 response_format：去掉 json_mode 重试一次
            if json_mode and "response_format" in str(e):
                try:
                    resp = await self._llm.ainvoke([to_langchain(m) for m in messages])
                except Exception as e2:  # noqa: BLE001
                    _record_llm_trace(self.model, success=False, duration_ms=int((time.monotonic() - _t0) * 1000), detail=str(e2)[:200])
                    raise LLMError(f"LLM 调用失败: {e2}") from e2
            else:
                _record_llm_trace(self.model, success=False, duration_ms=int((time.monotonic() - _t0) * 1000), detail=str(e)[:200])
                raise LLMError(f"LLM 调用失败: {e}") from e

        native = from_langchain(resp)
        finish_reason = (resp.response_metadata or {}).get("finish_reason")
        _record_llm_trace(self.model, success=True, duration_ms=int((time.monotonic() - _t0) * 1000))
        return LLMResponse(
            content=native.content or "",
            tool_calls=native.tool_calls or [],
            finish_reason=finish_reason,
            raw=resp,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ):
        """流式：langchain astream 逐 token 吐 content_delta；tool_calls 累积（覆盖式，与自研语义一致）。"""
        llm = self._bind(tools, json_mode)
        acc: Any = None
        async for chunk in llm.astream([to_langchain(m) for m in messages]):
            acc = chunk if acc is None else acc + chunk  # langchain __add__ 自动合并 tool_call_chunks
            tcs: list[ToolCall] = []
            for tc in getattr(acc, "tool_calls", None) or []:
                tcs.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        arguments=json.dumps(tc.get("args", {}), ensure_ascii=False),
                    )
                )
            fr = (chunk.response_metadata or {}).get("finish_reason")
            yield StreamChunk(content_delta=chunk.content or "", tool_calls=tcs, finish_reason=fr)
        if acc is not None:  # 确保结束时给出完整 tool_calls 与结束原因
            tcs = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=json.dumps(tc.get("args", {}), ensure_ascii=False),
                )
                for tc in getattr(acc, "tool_calls", None) or []
            ]
            fr = (acc.response_metadata or {}).get("finish_reason")
            yield StreamChunk(content_delta="", tool_calls=tcs, finish_reason=fr or "stop")
