"""Tool Executor：统一工具执行运行时。

从 ResearchAgent._run_tool 抽取，职责：
- 参数校验（委托 registry 的 execute，参数错误 retryable=False 不重试）
- 超时控制（asyncio.wait_for，默认 90s）
- 执行类错误退避重试（max_retries，默认 1 次）
- 连续失败检测（同一工具连续失败 N 次 → 标记 skipped，反馈模型换策略）
- 重复调用检测（同一工具 + 相同参数重复调用 → 提示循环，避免死循环）
- 空结果检测（success 但无实质内容 → 明确提示）
- 结果大小截断（max_chars）
- 执行 Trace 记录

用法：
    executor = ToolExecutor(registry, timeout=90, max_retries=1)
    text = await executor.execute("calculator", '{"expr": "1+1"}')
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from app.core.logging import get_logger
from app.tools.base import ToolRegistry, ToolResult

logger = get_logger("tools.executor")


def _has_meaningful_content(text: str) -> bool:
    """判断工具返回文本是否有实质内容。"""
    t = (text or "").strip()
    if not t:
        return False
    if t in ("[]", "{}", "None", "null", "无", "没有", "—"):
        return False
    return True


def _record_tool_trace(
    tool_name: str, *, success: bool, duration_ms: int, detail: str = ""
) -> None:
    try:
        from app.llm.trace import get_trace_ledger

        get_trace_ledger().record(
            "tool", tool_name, success=success, duration_ms=duration_ms, detail=detail
        )
    except Exception:  # noqa: BLE001
        pass


class ToolExecutor:
    """统一工具执行运行时（Agent 与 Planner 共用）。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout: float = 90.0,
        max_retries: int = 1,
        max_chars: int = 8000,
        consecutive_fail_limit: int = 2,
        duplicate_call_limit: int = 2,
    ) -> None:
        self.registry = registry
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.consecutive_fail_limit = consecutive_fail_limit
        self.duplicate_call_limit = duplicate_call_limit
        # 状态跟踪（每个 executor 实例独立）
        self._consecutive_failures: dict[str, int] = {}
        self._call_history: dict[str, list[str]] = {}  # tool -> [args_hash...]

    # ---------- 公开入口 ----------
    async def execute(self, tool_name: str, arguments_json: str) -> str:
        """执行工具并返回可回写给 LLM 的文本（含重试/超时/护栏/追踪）。"""
        # 兼容两种注册表形态：标准 ToolRegistry（call 接口）与 mock（无 get 时）
        if not hasattr(self.registry, "get"):
            return await self._execute_via_call(tool_name, arguments_json)
        tool = self.registry.get(tool_name)
        if tool is None:
            _record_tool_trace(tool_name, success=False, duration_ms=0, detail="tool_not_found")
            return f"工具 {tool_name} 不存在。请从可用工具列表中选择，或基于已有信息回答。"

        # 重复调用检测：同一工具 + 相同参数
        dup = self._check_duplicate_call(tool_name, arguments_json)
        if dup is not None:
            return dup

        # 连续失败检测：达到上限直接跳过并反馈
        if self._consecutive_failures.get(tool_name, 0) >= self.consecutive_fail_limit:
            return (
                f"工具 {tool_name} 已连续失败 {self.consecutive_fail_limit} 次，系统自动跳过。"
                "请换一个工具、修正参数，或基于已有信息直接回答。"
            )

        last_err = ""
        result: ToolResult | None = None
        _t0 = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(arguments_json),
                    timeout=self.timeout,
                )
                # 空结果检测
                if getattr(result, "success", True):
                    raw_output = getattr(result, "output", None)
                    empty = (
                        not _has_meaningful_content(result.to_message_content())
                        if raw_output is None
                        else not _has_meaningful_content(str(raw_output))
                    )
                else:
                    empty = False
                if empty:
                    _record_tool_trace(
                        tool_name, success=True,
                        duration_ms=int((time.monotonic() - _t0) * 1000),
                        detail="empty_result",
                    )
                    self._mark_success(tool_name)
                    return (
                        f"工具 {tool_name} 返回了空结果（无实质内容）。"
                        "可能是没有匹配的数据，请换一个查询词、换其他工具，或基于已有信息回答。"
                    )
                content = result.to_message_content()
                if attempt > 0:
                    content = f"[第 {attempt} 次重试后成功] {content}"
                # 结果截断
                if self.max_chars and len(content) > self.max_chars:
                    content = content[: self.max_chars] + "\n…(结果过长已截断)"
                _record_tool_trace(
                    tool_name, success=True,
                    duration_ms=int((time.monotonic() - _t0) * 1000),
                    detail=f"attempt={attempt + 1}",
                )
                self._mark_success(tool_name)
                return content
            except asyncio.TimeoutError:
                last_err = f"第 {attempt + 1} 次执行超时（>{self.timeout}s）"
                logger.warning("工具 %s %s", tool_name, last_err)
            except Exception as e:  # noqa: BLE001
                last_err = f"第 {attempt + 1} 次执行失败：{e}"
                logger.warning("工具 %s %s", tool_name, last_err)
            # 不可重试的错误（参数类）：终止重试循环
            if result is not None and not getattr(result, "retryable", True):
                break
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))  # 退避：0.5s / 1s

        self._mark_failure(tool_name)
        _record_tool_trace(
            tool_name, success=False,
            duration_ms=int((time.monotonic() - _t0) * 1000),
            detail=last_err[:200],
        )
        return (
            f"工具 {tool_name} 执行失败（{last_err}）。"
            "请检查参数是否正确（例如换一个更精确的查询词），"
            "或尝试其他工具，或基于已有信息继续。"
        )

    async def _execute_via_call(self, tool_name: str, arguments_json: str) -> str:
        """兼容旧接口：注册表只有 call()（如测试 mock），走 call 路径。

        与 ResearchAgent 旧 _run_tool 行为一致（call 内部自带参数校验/错误归一化）。
        """
        last_err = ""
        result: ToolResult | None = None
        _t0 = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.registry.call(tool_name, arguments_json),
                    timeout=self.timeout,
                )
                if getattr(result, "success", True):
                    raw_output = getattr(result, "output", None)
                    empty = (
                        not _has_meaningful_content(result.to_message_content())
                        if raw_output is None
                        else not _has_meaningful_content(str(raw_output))
                    )
                else:
                    empty = False
                if empty:
                    _record_tool_trace(
                        tool_name, success=True,
                        duration_ms=int((time.monotonic() - _t0) * 1000),
                        detail="empty_result",
                    )
                    self._mark_success(tool_name)
                    return (
                        f"工具 {tool_name} 返回了空结果（无实质内容）。"
                        "可能是没有匹配的数据，请换一个查询词、换其他工具，或基于已有信息回答。"
                    )
                content = result.to_message_content()
                if attempt > 0:
                    content = f"[第 {attempt} 次重试后成功] {content}"
                if self.max_chars and len(content) > self.max_chars:
                    content = content[: self.max_chars] + "\n…(结果过长已截断)"
                _record_tool_trace(
                    tool_name, success=True,
                    duration_ms=int((time.monotonic() - _t0) * 1000),
                    detail=f"attempt={attempt + 1}",
                )
                self._mark_success(tool_name)
                return content
            except asyncio.TimeoutError:
                last_err = f"第 {attempt + 1} 次执行超时（>{self.timeout}s）"
                logger.warning("工具 %s %s", tool_name, last_err)
            except Exception as e:  # noqa: BLE001
                last_err = f"第 {attempt + 1} 次执行失败：{e}"
                logger.warning("工具 %s %s", tool_name, last_err)
            if result is not None and not getattr(result, "retryable", True):
                break
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
        self._mark_failure(tool_name)
        _record_tool_trace(
            tool_name, success=False,
            duration_ms=int((time.monotonic() - _t0) * 1000),
            detail=last_err[:200],
        )
        return (
            f"工具 {tool_name} 执行失败（{last_err}）。"
            "请检查参数是否正确（例如换一个更精确的查询词），"
            "或尝试其他工具，或基于已有信息继续。"
        )

    # ---------- 内部状态 ----------
    def _mark_success(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] = 0

    def _mark_failure(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] = (
            self._consecutive_failures.get(tool_name, 0) + 1
        )

    def _check_duplicate_call(self, tool_name: str, arguments_json: str) -> str | None:
        """重复调用检测：同工具同参数连续调用达到上限时提示循环。"""
        try:
            args = json.loads(arguments_json or "{}")
            arg_hash = hashlib.sha1(
                json.dumps(args, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:12]
        except json.JSONDecodeError:
            return None
        history = self._call_history.setdefault(tool_name, [])
        history.append(arg_hash)
        # 只保留最近 N 条
        if len(history) > self.duplicate_call_limit + 1:
            history.pop(0)
        recent_same = sum(1 for h in history if h == arg_hash)
        # 阈值语义：limit=2 表示「同参数出现 3 次（含首次）才提示」，避免首两次正常调用被误伤
        if recent_same > self.duplicate_call_limit:
            return (
                f"⚠️ 检测到工具 {tool_name} 使用相同参数重复调用 {recent_same} 次，"
                "可能陷入循环。请停止重复调用，换一种方式（修改参数/换工具/直接回答）。"
            )
        return None

    def reset_state(self) -> None:
        """重置失败计数与调用历史（新会话时调用）。"""
        self._consecutive_failures.clear()
        self._call_history.clear()
