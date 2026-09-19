"""LLM Gateway：统一模型调用入口。

业务代码不应直接调用 get_llm()/get_aux_llm()，而是通过 gateway：

    from app.llm.gateway import llm_gateway
    resp = await llm_gateway.chat(messages, task_type=TaskType.QUERY_REWRITE)

职责：
- 按任务类型自动路由主/辅模型（ModelRouter）
- 统一记录 usage（token/cost）与 trace（latency/success/model）
- 统一超时与失败处理（可选：由调用方决定是否重试）
- 支持显式覆盖（force_main=True 强制主模型；model= 指定模型）
"""
from __future__ import annotations

import time
from typing import Any

from app.llm.base import BaseLLM, ChatMessage
from app.llm.router import TaskType, get_model_router


class LLMGateway:
    """模型调用门面。"""

    def __init__(self) -> None:
        self._router = get_model_router()
        self._aux_open_until = 0.0
        self._aux_cooldown_seconds = 60.0

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        task_type: str | TaskType | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        force_main: bool = False,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> Any:
        """统一调用：路由 → 实例化 → 调用 → 记录。返回 LLMResponse。"""
        from app.llm.factory import get_aux_llm, get_llm

        decision = self._router.route(task_type, force_main=force_main)
        if decision["use_aux"] and time.monotonic() < self._aux_open_until:
            decision["use_aux"] = False
            decision["circuit_fallback"] = True
        t0 = time.monotonic()
        # 实例选择：显式 model 覆盖 > 路由决策
        if model:
            llm: BaseLLM = get_llm(
                provider=provider, model=model, base_url=base_url, api_key=api_key
            )
        elif decision["use_aux"]:
            llm = get_aux_llm()
        else:
            llm = get_llm()

        try:
            resp = await llm.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except Exception:
            # 辅助模型限流/断连时当次立即回退主模型，并短时熔断，
            # 避免每个内部任务都重复等待同一个故障上游。
            if not decision["use_aux"] or model:
                raise
            self._aux_open_until = time.monotonic() + self._aux_cooldown_seconds
            decision["use_aux"] = False
            decision["circuit_fallback"] = True
            llm = get_llm()
            resp = await llm.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        # usage 与 trace 由底层 provider 记录（openai_provider 已接入 UsageLedger/TraceLedger），
        # 这里补充任务类型标注，便于按任务维度分析成本。
        try:
            from app.llm.trace import get_trace_ledger

            get_trace_ledger().record(
                "model_router",
                decision["task_type"],
                success=True,
                duration_ms=int((time.monotonic() - t0) * 1000),
                detail=f"use_aux={decision['use_aux']} model={getattr(llm, 'model', '?')}",
            )
        except Exception:  # noqa: BLE001
            pass
        return resp

    async def chat_json(
        self,
        prompt: str,
        *,
        task_type: str | TaskType | None = None,
        validator=None,
        max_retries: int = 1,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        force_main: bool = False,
    ) -> tuple[Any, str | None]:
        """结构化输出便捷入口：prompt → JSON 解析 + 校验，失败按 max_retries 重试。

        返回 (data, error)：成功时 error=None；重试耗尽后 error 非空。
        """
        from app.llm.base import ChatMessage, MessageRole
        from app.llm.structured_output import parse_and_validate

        last_err: str | None = None
        for attempt in range(max_retries + 1):
            resp = await self.chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                task_type=task_type,
                json_mode=True,
                temperature=temperature,
                max_tokens=max_tokens,
                force_main=force_main,
            )
            data, err = parse_and_validate(resp.content or "", validator)
            if err is None:
                return data, None
            last_err = err
        return None, last_err or "结构化输出失败"

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        task_type: str | TaskType | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        force_main: bool = False,
        model: str | None = None,
    ):
        """流式调用（async generator）：路由 → 实例化 → 流式转发。

        Yield 与底层 BaseLLM.stream 相同的 chunk 对象（StreamChunk）。
        """
        from app.llm.factory import get_aux_llm, get_llm

        decision = self._router.route(task_type, force_main=force_main)
        t0 = time.monotonic()
        if model:
            llm: BaseLLM = get_llm(model=model)
        elif decision["use_aux"]:
            llm = get_aux_llm()
        else:
            llm = get_llm()

        async for chunk in llm.stream(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            yield chunk

        # 流结束后补记任务类型 Trace（provider 已记 usage/基础 trace）
        try:
            from app.llm.trace import get_trace_ledger

            get_trace_ledger().record(
                "model_router",
                decision["task_type"],
                success=True,
                duration_ms=int((time.monotonic() - t0) * 1000),
                detail=f"use_aux={decision['use_aux']} model={getattr(llm, 'model', '?')} stream=1",
            )
        except Exception:  # noqa: BLE001
            pass


# 全局单例
_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


def reset_llm_gateway() -> None:
    global _gateway
    _gateway = None


# 便捷别名（保持与 factory 风格一致）
llm_gateway = get_llm_gateway()
