"""OpenAI 兼容 LLM 实现。

支持任意遵循 OpenAI Chat Completions 协议的端点（OpenAI / DeepSeek / 本地 Ollama / vLLM 等）。
内置：
- `tenacity` 指数退避重试
- 工具调用（tool_calls）解析
- 流式 token 产出
- 会话级链路追踪（Trace）
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.core.exceptions import LLMError
from app.llm.base import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    MessageRole,
    StreamChunk,
    ToolCall,
)
from app.llm.usage import get_usage_ledger


def _record_llm_trace(
    model: str, *, success: bool, duration_ms: int, detail: str = "", channel: dict | None = None
) -> None:
    """记录一次 LLM 调用到链路追踪（失败静默，不干扰主流程）。"""
    try:
        from app.llm.trace import get_trace_ledger

        channel = channel or {}
        channel_detail = (
            f"channel={channel.get('channel', 'unknown')} "
            f"endpoint={channel.get('endpoint', '')} "
        )
        get_trace_ledger().record(
            "llm", model, success=success, duration_ms=duration_ms,
            detail=channel_detail + detail,
        )
    except Exception:  # noqa: BLE001
        pass


class OpenAICompatibleLLM(BaseLLM):
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
        s = get_settings().llm
        self.model = model or s.model
        self.temperature = s.temperature if temperature is None else temperature
        self.max_tokens = s.max_tokens if max_tokens is None else max_tokens
        self.timeout = s.timeout if timeout is None else timeout
        self.base_url = base_url or s.base_url
        from app.llm.channel import classify_channel

        self.channel_info = classify_channel(self.base_url)
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key or s.api_key or "EMPTY",
            timeout=self.timeout,
            max_retries=0,  # 退避由下方装饰器统一控制
        )

    def _payload(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        _t0 = time.monotonic()
        try:
            resp = await self.client.chat.completions.create(
                **self._payload(messages, tools, temperature, max_tokens, json_mode)
            )
        except Exception as e:  # noqa: BLE001
            # 部分供应商/本地模型不支持 response_format：去掉 json_mode 重试一次
            if json_mode and "response_format" in str(e):
                try:
                    resp = await self.client.chat.completions.create(
                        **self._payload(messages, tools, temperature, max_tokens, False)
                    )
                except Exception as e2:  # noqa: BLE001
                    _record_llm_trace(self.model, success=False, duration_ms=int((time.monotonic() - _t0) * 1000), detail=str(e2)[:200], channel=self.channel_info)
                    raise LLMError(f"LLM 调用失败: {e2}") from e2
            else:
                _record_llm_trace(self.model, success=False, duration_ms=int((time.monotonic() - _t0) * 1000), detail=str(e)[:200], channel=self.channel_info)
                raise LLMError(f"LLM 调用失败: {e}") from e

        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
            )

        # 成本 / token 账本：每次调用后累计用量
        usage = getattr(resp, "usage", None)
        if usage is not None:
            get_usage_ledger().record(
                self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
            _record_llm_trace(
                self.model,
                success=True,
                duration_ms=int((time.monotonic() - _t0) * 1000),
                detail=(
                    f"prompt={getattr(usage, 'prompt_tokens', 0)} "
                    f"completion={getattr(usage, 'completion_tokens', 0)}"
                ), channel=self.channel_info,
            )

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=resp.choices[0].finish_reason,
            raw=resp,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                **self._payload(messages, tools, temperature, max_tokens),
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式补全（含工具调用增量）。

        OpenAI 流式 API 中 tool_calls 以增量片段到达（每片有 index/id/name/arguments
        片段），按 index 合并累积，流结束时得到完整 tool_calls。
        """
        _t0 = time.monotonic()
        try:
            stream = await self.client.chat.completions.create(
                **self._payload(messages, tools, temperature, max_tokens),
                stream=True,
            )
        except Exception as e:  # noqa: BLE001
            _record_llm_trace(self.model, success=False, duration_ms=int((time.monotonic() - _t0) * 1000), detail=str(e)[:200], channel=self.channel_info)
            raise LLMError(f"LLM 流式调用失败: {e}") from e

        # tool_calls 累积表: index -> {id, name, arguments}
        acc: dict[int, dict[str, str]] = {}
        finish: str | None = None
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish = choice.finish_reason
            if delta and delta.content:
                yield StreamChunk(content_delta=delta.content)
            # 工具调用增量：按 index 合并
            for tc in delta.tool_calls or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        # 流结束：汇总完整 tool_calls（若有）
        tool_calls: list[ToolCall] | None = None
        if acc:
            tool_calls = [
                ToolCall(
                    id=v["id"] or f"call_{i}",
                    name=v["name"],
                    arguments=v["arguments"] or "{}",
                )
                for i, v in sorted(acc.items())
            ]
        yield StreamChunk(content_delta="", tool_calls=tool_calls, finish_reason=finish or "stop")
        # 流式调用完成（成功）：记录 trace（含流式用量）
        _record_llm_trace(
            self.model,
            success=True,
            duration_ms=int((time.monotonic() - _t0) * 1000),
            detail=f"stream finish={finish or 'stop'} tool_calls={len(tool_calls or [])}",
            channel=self.channel_info,
        )
