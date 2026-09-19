"""LLM 抽象接口。

设计目标：
- 与具体供应商解耦，任何 OpenAI 兼容 / 自研模型都可接入。
- 同时支持「同步补全」与「流式补全」（以 async generator 吐 token）。
- 工具调用（function calling）以 OpenAI 风格的 `tool_calls` 表示，便于 Agent 解析。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型请求调用某个工具的意图。"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class ChatMessage:
    role: MessageRole
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # tool 角色消息回写对应调用 id
    name: str | None = None
    images: list[str] | None = None  # 多模态：图片 data URL 列表（base64），仅当轮 user 消息有效

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value}
        if self.images and self.role == MessageRole.USER:
            # OpenAI 兼容多模态格式：content 为 [text, image_url...] 数组
            parts: list[dict] = []
            if self.content:
                parts.append({"type": "text", "text": self.content})
            for img in self.images:
                parts.append({"type": "image_url", "image_url": {"url": img}})
            d["content"] = parts
        else:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMessage":
        """从 to_dict 的输出完整恢复（含 tool_calls），用于持久化后端。"""
        tcs: list[ToolCall] | None = None
        raw_tcs = d.get("tool_calls")
        if raw_tcs:
            tcs = []
            for tc in raw_tcs:
                fn = tc.get("function", {})
                tcs.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", "{}"),
                    )
                )
        return cls(
            role=MessageRole(d.get("role", "user")),
            content=d.get("content", "") or "",
            tool_calls=tcs,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: Any = None


@dataclass
class StreamChunk:
    """流式补全的增量块。

    - content_delta: 文本增量（可能为空字符串）
    - tool_calls: 本轮流式过程中累积的工具调用（增量合并后，流结束时完整）
    - finish_reason: 结束原因（stop / tool_calls / length 等），None 表示未结束
    """

    content_delta: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


class BaseLLM(abc.ABC):
    """所有 LLM 供应商需实现的接口。"""

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """同步（一次性）补全。json_mode=True 时请求结构化 JSON 输出（供应商不支持则忽略）。"""

    @abc.abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """流式补全，逐 token 产出字符串（纯文本场景）。"""

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式补全（含工具调用增量）。

        默认实现：逐 token 产出 content_delta，流结束时根据 stream() 无法获取的
        tool_calls 信息退化为空 tool_calls（子类应覆写以支持工具流式）。
        返回的最后一帧带 finish_reason。
        """
        collected: list[str] = []
        async for token in self.stream(messages, tools=tools, temperature=temperature, max_tokens=max_tokens):
            collected.append(token)
            yield StreamChunk(content_delta=token)
        yield StreamChunk(content_delta="", tool_calls=None, finish_reason="stop")
