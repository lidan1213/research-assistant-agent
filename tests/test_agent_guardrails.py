"""Agent 护栏单元测试：token 预算守门 / observation 截断 / 工具失败自修正。

使用轻量 Fake 组件，不依赖真实 LLM 或网络。
"""
from __future__ import annotations

import asyncio
from typing import List

from app.agent.agent import ResearchAgent
from app.llm.base import ChatMessage, LLMResponse, MessageRole, ToolCall
from app.tools.base import ToolResult


# ---------------- Fake 组件 ----------------
class FakeLLM:
    def __init__(self, responses, force_finish: str = "（强制收尾）"):
        self.responses = responses
        self.force_finish = force_finish
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        if messages and "直接给出最终回答" in (messages[-1].content or ""):
            return LLMResponse(content=self.force_finish, tool_calls=[])
        r = self.responses[self.calls]
        self.calls += 1
        return r

    async def chat_stream(self, messages, tools=None, **kwargs):
        """流式版：复用 chat 结果，分块产出（最后一块带 tool_calls/finish_reason）。"""
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

    async def add_user(self, sid, m):
        self.log.append(("u", m))

    async def add_assistant(self, sid, m, tool_calls=None):
        self.log.append(("a", m))

    async def add_tool(self, sid, tid, name, obs):
        self.log.append(("t", name, obs))

    async def history(self, sid):
        return self._history


class FakeToolResult:
    def __init__(self, content: str):
        self._c = content

    def to_message_content(self):
        return self._c


class OkTools:
    def __init__(self, content: str = "工具结果"):
        self.content = content

    def schemas(self):
        return []

    async def call(self, name, args):
        return FakeToolResult(self.content)


class FlakyTools:
    def __init__(self, fail: int = 1):
        self.fail = fail
        self.calls = 0

    def schemas(self):
        return []

    async def call(self, name, args):
        self.calls += 1
        if self.calls <= self.fail:
            raise RuntimeError("boom")
        return FakeToolResult("recovered")


def _tool_response(name: str = "search"):
    return LLMResponse(
        content="let me use a tool",
        tool_calls=[ToolCall(id="t1", name=name, arguments="{}")],
    )


def _answer_response(text: str = "完成"):
    return LLMResponse(content=text, tool_calls=[])


async def _collect(agent: ResearchAgent, msg: str, **kw) -> List:
    return [e async for e in agent.stream("sid", msg, **kw)]


# ---------------- 测试 ----------------
def test_token_budget_forces_finish():
    mem = FakeMemory(history=[ChatMessage(role=MessageRole.USER, content="x" * 200000)])
    llm = FakeLLM([_tool_response(), _answer_response()], force_finish="预算收尾答案")
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(), planner=None,
        token_budget=100, max_iterations=8,
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    answer = [e for e in events if e.type == "answer"][0]
    assert answer.data["content"] == "预算收尾答案"
    assert answer.data.get("reason") == "token_budget"


def test_observation_truncation():
    mem = FakeMemory()
    llm = FakeLLM([_tool_response(), _answer_response("完成")])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(content="A" * 10000), planner=None,
        max_obs_chars=100, token_budget=10**9,
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    obs = [e for e in events if e.type == "observation"][0]
    assert obs.data["truncated"] is True
    assert len(obs.data["output"]) <= 100 + 40


def test_tool_failure_self_correction():
    mem = FakeMemory()
    llm = FakeLLM([_tool_response(), _answer_response("完成")])
    tools = FlakyTools(fail=1)
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=tools, planner=None,
        tool_max_retries=1, token_budget=10**9,
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    assert tools.calls == 2  # 失败 1 次 + 重试用 1 次
    obs = [e for e in events if e.type == "observation"][0]
    assert "重试后成功" in obs.data["output"]


# ---------------- ToolGuard：空结果提示 / 不可重试错误不重试 ----------------

class EmptyResultTools:
    """工具返回空内容（success 但无实质内容）。"""

    def schemas(self):
        return []

    async def call(self, name, args):
        return ToolResult(success=True, output="")


class ParamErrorTools:
    """工具返回参数类错误（retryable=False）。"""

    def __init__(self):
        self.calls = 0

    def schemas(self):
        return []

    async def call(self, name, args):
        self.calls += 1
        return ToolResult(success=False, error="参数校验失败: bad", retryable=False)


def test_tool_empty_result_hint():
    """空结果提示：success 但无内容时，observation 明示无数据，避免 LLM 误判。"""
    mem = FakeMemory()
    llm = FakeLLM([_tool_response(), _answer_response("完成")])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=EmptyResultTools(), planner=None,
        tool_max_retries=1, token_budget=10**9,
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    obs = [e for e in events if e.type == "observation"][0]
    assert "空结果" in obs.data["output"]


def test_param_error_no_retry():
    """不可重试错误：参数类失败不重试（只调用 1 次，避免白费重试）。"""
    mem = FakeMemory()
    llm = FakeLLM([_tool_response(), _answer_response("完成")])
    tools = ParamErrorTools()
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=tools, planner=None,
        tool_max_retries=3, token_budget=10**9,  # 即使配置 3 次重试也不触发
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    assert tools.calls == 1  # 参数错误不重试
    obs = [e for e in events if e.type == "observation"][0]
    assert "参数校验失败" in obs.data["output"]


# ---------------- 上下文压缩（ContextManager） ----------------

def test_history_compression():
    """历史超阈值：早期轮次被压缩为摘要，保留最近原文。"""
    mem = FakeMemory(history=[
        ChatMessage(role=MessageRole.USER, content="早期问题" + "a" * 80),
        ChatMessage(role=MessageRole.ASSISTANT, content="早期回答" + "b" * 80),
        ChatMessage(role=MessageRole.USER, content="早期问题2" + "c" * 80),
        ChatMessage(role=MessageRole.ASSISTANT, content="早期回答2" + "d" * 80),
        ChatMessage(role=MessageRole.USER, content="中期问题" + "e" * 80),
        ChatMessage(role=MessageRole.ASSISTANT, content="中期回答" + "f" * 80),
        ChatMessage(role=MessageRole.USER, content="最近问题" + "g" * 80),
        ChatMessage(role=MessageRole.ASSISTANT, content="最近回答" + "h" * 80),
    ])
    llm = FakeLLM([_answer_response("（摘要）用户在研究量子点，已确认带隙可调。")])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(), planner=None,
        token_budget=200,  # 8 条 × ~40 token = 320 > 200×0.55=110 → 触发压缩
        history_keep_msgs=4,  # 保留最近 4 条原文，更早的压成摘要
    )
    # 直接调用压缩方法验证
    compressed = asyncio.run(agent._maybe_compress_history("sid", mem._history))
    # 压缩后首条是摘要 system 消息
    assert compressed[0].role == MessageRole.SYSTEM
    assert "早期对话摘要" in (compressed[0].content or "")
    # 保留最近原文（最后一条仍是最近回答）
    assert (compressed[-1].content or "").startswith("最近回答")


def test_history_no_compress_below_threshold():
    """历史未超阈值：不触发压缩，原样返回。"""
    mem = FakeMemory(history=[
        ChatMessage(role=MessageRole.USER, content="短问题"),
        ChatMessage(role=MessageRole.ASSISTANT, content="短回答"),
    ])
    llm = FakeLLM([_answer_response("x")])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(), planner=None, token_budget=10**9,
    )
    result = asyncio.run(agent._maybe_compress_history("sid", mem._history))
    assert len(result) == 2
    assert result[0].role == MessageRole.USER  # 未插入摘要消息


# ---------------- 空答案自愈重试 ----------------

def test_empty_answer_retry():
    """LLM 首轮返回空内容：自动重试一次并产出最终答案。"""
    mem = FakeMemory()
    # 第一个响应：空内容无工具调用 -> 触发重试；第二个：正常答案
    llm = FakeLLM([
        LLMResponse(content="", tool_calls=[], finish_reason="stop"),
        _answer_response("重试后的答案"),
    ])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(), planner=None, token_budget=10**9,
    )
    events = asyncio.run(_collect(agent, "hi", use_plan=False))
    answer = [e for e in events if e.type == "answer"][0]
    assert answer.data["content"] == "重试后的答案"
    assert llm.calls >= 2  # 确实发生了重试


# ---------------- LLM 超时降级 / 无进展循环保护 ----------------

class PartialStreamFailureLLM(FakeLLM):
    async def chat_stream(self, messages, tools=None, **kwargs):
        from app.llm.base import StreamChunk

        yield StreamChunk(content_delta="已经生成的有效结论")
        raise TimeoutError("stream stalled")


def test_partial_stream_timeout_keeps_generated_answer():
    """流末尾超时时保留已生成内容，不能把整条回答判成失败。"""
    mem = FakeMemory()
    llm = PartialStreamFailureLLM([])
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(), planner=None,
        token_budget=10**9, llm_recovery_timeout=0.1,
    )
    events = asyncio.run(_collect(agent, "普通复杂问题", use_plan=False))
    answer = [e for e in events if e.type == "answer"][0]
    assert "已经生成的有效结论" in answer.data["content"]
    assert answer.data["reason"] == "partial_stream_recovery"
    assert not [e for e in events if e.type == "error"]


def test_repeated_observation_forces_early_finish():
    """参数变化但工具结果不变时，按无进展而非最大迭代次数提前收尾。"""
    mem = FakeMemory()
    llm = FakeLLM([
        LLMResponse(content="", tool_calls=[ToolCall(id="t1", name="search", arguments='{"q":"a"}')]),
        LLMResponse(content="", tool_calls=[ToolCall(id="t2", name="search", arguments='{"q":"b"}')]),
    ], force_finish="基于已有信息收尾")
    agent = ResearchAgent(
        llm=llm, memory=mem, tools=OkTools(content="相同结果"), planner=None,
        token_budget=10**9, max_iterations=8, no_progress_limit=1,
    )
    events = asyncio.run(_collect(agent, "需要多步搜索的问题", use_plan=False))
    answer = [e for e in events if e.type == "answer"][0]
    assert answer.data["reason"] == "no_progress"
    assert answer.data["content"] == "基于已有信息收尾"
    assert llm.calls == 2
