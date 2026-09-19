"""Agent 增强测试：工具并行执行 / 会话自动摘要 / 答案质量自评。"""
from __future__ import annotations

import asyncio
import time

from app.agent.agent import ResearchAgent
from app.llm.base import ChatMessage, LLMResponse, MessageRole, ToolCall
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    async def chat_stream(self, messages, tools=None, **kwargs):
        from app.llm.base import StreamChunk

        resp = await self.chat(messages, tools=tools, **kwargs)
        content = resp.content or ""
        step = max(1, len(content) // 3)
        for i in range(0, len(content), step):
            yield StreamChunk(content_delta=content[i:i + step])
        yield StreamChunk(
            content_delta="",
            tool_calls=resp.tool_calls or None,
            finish_reason=resp.finish_reason or ("tool_calls" if resp.tool_calls else "stop"),
        )


class FakeMemory:
    def __init__(self, history=None):
        self._history = history or []
        self.log = []
        self.summary = None

    async def add_user(self, sid, m):
        self.log.append(("u", m))

    async def add_assistant(self, sid, m, tool_calls=None):
        self.log.append(("a", m))

    async def add_tool(self, sid, tid, name, obs):
        self.log.append(("t", name, obs))

    async def history(self, sid):
        return self._history

    async def set_summary(self, sid, summary):
        self.summary = summary


class ParallelTools:
    """记录并发度的工具：并行时总耗时 ≈ 单工具耗时（而非 3 倍）。"""

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def schemas(self):
        return []

    async def call(self, name, args):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return ToolResult(success=True, output=f"{name} done")


def _multi_tool_response():
    """返回 3 个并行工具调用。"""
    return LLMResponse(
        content="并行搜索三个方向",
        tool_calls=[
            ToolCall(id="c1", name="s1", arguments="{}"),
            ToolCall(id="c2", name="s2", arguments="{}"),
            ToolCall(id="c3", name="s3", arguments="{}"),
        ],
        finish_reason="tool_calls",
    )


async def _collect(agent, msg):
    return [e async for e in agent.stream("sid", msg, use_plan=False)]


def test_tool_parallel_execution():
    """工具并行：3 个工具同时执行，最大并发=3，总耗时远小于串行 3×delay。"""
    llm = FakeLLM([_multi_tool_response(), LLMResponse(content="完成", tool_calls=[])])
    tools = ParallelTools(delay=0.3)
    agent = ResearchAgent(llm=llm, memory=FakeMemory(), tools=tools, planner=None, token_budget=10**9)

    t0 = time.monotonic()
    events = asyncio.run(_collect(agent, "hi"))
    elapsed = time.monotonic() - t0

    assert tools.max_active == 3  # 三个工具真正并行
    assert elapsed < 0.7  # 串行会要 0.9s+，并行约 0.3s（留裕量）
    obs = [e for e in events if e.type == "observation"]
    assert len(obs) == 3
    # observation 按原始顺序回写
    assert obs[0].data["tool"] == "s1"
    assert obs[2].data["tool"] == "s3"


def test_session_summary_generated():
    """会话结束：自动生成摘要并写入 memory。"""
    mem = FakeMemory(history=[
        ChatMessage(role=MessageRole.USER, content="帮我调研量子点太阳能电池"),
        ChatMessage(role=MessageRole.ASSISTANT, content="量子点电池效率已达 30%，主要瓶颈是稳定性"),
    ])
    # 调用顺序：主回答(calls0) → 摘要(calls1，facts 因 longterm=None 跳过)
    llm = FakeLLM([
        LLMResponse(content="完成", tool_calls=[]),
        LLMResponse(content="用户研究量子点太阳能电池，结论：效率30%，瓶颈为稳定性。"),
    ])
    agent = ResearchAgent(llm=llm, memory=mem, tools=ParallelTools(), planner=None, token_budget=10**9)
    asyncio.run(_collect(agent, "帮我调研量子点太阳能电池"))
    assert mem.summary is not None
    assert "量子点" in mem.summary


def test_self_eval_callback_invoked():
    """答案自评：low confidence 时触发回调，携带 tooltip。"""
    mem = FakeMemory()
    evals = []

    async def cb(_sid, data):
        evals.append(data)

    # 调用顺序：主回答(calls0，>=10字符) → 自评(calls1；摘要因 history 空跳过、facts 因 longterm=None 跳过)
    llm = FakeLLM([
        LLMResponse(content="根据相对论，光速约为每秒 299792458 米。", tool_calls=[]),
        LLMResponse(content='{"answered": true, "confidence": "low", "missing": "缺乏文献支撑"}'),
    ])
    agent = ResearchAgent(llm=llm, memory=mem, tools=ParallelTools(), planner=None, token_budget=10**9)
    agent._eval_callback = cb
    asyncio.run(_collect(agent, "光速是多少"))
    assert len(evals) == 1
    assert evals[0]["confidence"] == "low"
    assert "置信度较低" in evals[0]["tooltip"]
