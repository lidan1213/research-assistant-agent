"""Tool Executor 测试：超时重试/参数错误不重试/连续失败/重复调用/截断。"""
from __future__ import annotations

import asyncio

import pytest

from app.tools.base import FunctionTool, ToolRegistry, ToolResult
from app.tools.executor import ToolExecutor


class StubTool:
    """模拟工具：可配置抛异常/耗时/参数校验（重试针对抛异常，与旧行为一致）。"""

    def __init__(
        self,
        name: str = "stub",
        *,
        raise_exc: bool = False,
        fail_times: int = 0,
        delay: float = 0,
        output: str = "ok",
    ):
        self.name = name
        self.description = "stub tool"
        self._raise_exc = raise_exc
        self._fail_times = fail_times
        self._delay = delay
        self._output = output
        self.calls = 0

    def to_function_schema(self):
        return {"name": self.name, "description": self.description, "parameters": {}}

    async def execute(self, arguments_json: str) -> ToolResult:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise_exc and self.calls <= self._fail_times:
            raise RuntimeError("boom")
        if not self._raise_exc and self._fail_times and self.calls <= self._fail_times:
            return ToolResult(success=False, error="boom", retryable=False)
        return ToolResult(success=True, output=self._output)


def _registry(tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


def test_execute_success():
    tool = StubTool()
    ex = ToolExecutor(_registry(tool), max_retries=1)
    out = asyncio.run(ex.execute("stub", "{}"))
    assert "ok" in out  # ToolResult 内容以 <data> 包裹
    assert tool.calls == 1


def test_execute_retries_transient_failure():
    """执行类异常应重试（工具抛异常，非 ToolResult 失败）。"""
    tool = StubTool(raise_exc=True, fail_times=1)
    ex = ToolExecutor(_registry(tool), max_retries=2)
    out = asyncio.run(ex.execute("stub", "{}"))
    assert tool.calls == 2  # 失败 1 次 + 成功 1 次
    assert "重试后成功" in out


def test_execute_no_retry_on_param_error():
    """参数类错误（ToolResult retryable=False）直接返回，不重试。"""
    tool = StubTool(fail_times=99)
    ex = ToolExecutor(_registry(tool), max_retries=3)
    out = asyncio.run(ex.execute("stub", "{}"))
    assert tool.calls == 1  # 只调用一次
    assert "执行失败" in out


def test_execute_timeout_retries():
    """超时应重试并最终失败。"""
    tool = StubTool(delay=0.3, raise_exc=True, fail_times=99)
    ex = ToolExecutor(_registry(tool), timeout=0.1, max_retries=1)
    out = asyncio.run(ex.execute("stub", "{}"))
    assert "超时" in out
    assert tool.calls == 2


def test_consecutive_failure_skip():
    """同一工具连续失败达到上限后自动跳过（用不同参数避免触发重复调用检测）。"""
    tool = StubTool(raise_exc=True, fail_times=99)
    ex = ToolExecutor(_registry(tool), max_retries=1, consecutive_fail_limit=2)
    # 第一次：失败（重试 1 次 = 2 calls）→ 连续失败 +1
    out1 = asyncio.run(ex.execute("stub", '{"n": 1}'))
    assert "执行失败" in out1
    # 第二次：失败 → 连续失败 2
    out2 = asyncio.run(ex.execute("stub", '{"n": 2}'))
    assert "执行失败" in out2
    # 第三次：直接跳过，不执行
    before = tool.calls
    out3 = asyncio.run(ex.execute("stub", '{"n": 3}'))
    assert "自动跳过" in out3
    assert tool.calls == before  # 未执行


def test_duplicate_call_detection():
    """同工具同参数重复调用达到上限时提示循环。"""
    tool = StubTool()
    ex = ToolExecutor(_registry(tool), duplicate_call_limit=2)
    for _ in range(2):
        out = asyncio.run(ex.execute("stub", '{"x": 1}'))
        assert "ok" in out
    out3 = asyncio.run(ex.execute("stub", '{"x": 1}'))
    assert "重复调用" in out3
    # 不同参数不受影响
    out4 = asyncio.run(ex.execute("stub", '{"x": 2}'))
    assert "ok" in out4


def test_result_truncation():
    tool = StubTool(output="x" * 1000)
    ex = ToolExecutor(_registry(tool), max_chars=100)
    out = asyncio.run(ex.execute("stub", "{}"))
    assert len(out) <= 120
    assert "截断" in out


def test_tool_not_found():
    ex = ToolExecutor(ToolRegistry())
    out = asyncio.run(ex.execute("nonexistent", "{}"))
    assert "不存在" in out


def test_empty_result_hint():
    tool = StubTool(output="[]")
    ex = ToolExecutor(_registry(tool))
    out = asyncio.run(ex.execute("stub", "{}"))
    assert "空结果" in out


def test_reset_state():
    tool = StubTool(raise_exc=True, fail_times=99)
    ex = ToolExecutor(_registry(tool), max_retries=1, consecutive_fail_limit=2)
    asyncio.run(ex.execute("stub", "{}"))
    asyncio.run(ex.execute("stub", "{}"))
    ex.reset_state()
    out = asyncio.run(ex.execute("stub", "{}"))
    assert "自动跳过" not in out  # 重置后重新计数，不是跳过
    assert "执行失败" in out
