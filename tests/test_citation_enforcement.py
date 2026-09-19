"""强制引用测试：使用检索工具后，最终答案必须带 [n] 引用标记。

验证三个行为：
1. 用过 knowledge_search 但答案缺 [n] → 追加引用规范让 LLM 重写一次；
2. 答案已带 [n] → 不触发重写；
3. 未使用检索工具 → 不强制引用。
"""
from __future__ import annotations

import asyncio

from app.agent.agent import ResearchAgent
from app.llm.base import LLMResponse, ToolCall
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.seen_system_messages: list[str] = []

    async def chat(self, messages, tools=None, **kwargs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        for m in messages:
            if getattr(m, "role", None) and m.role.value == "system":
                self.seen_system_messages.append(m.content or "")
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
    async def add_user(self, sid, m):
        pass

    async def add_assistant(self, sid, m, tool_calls=None):
        pass

    async def add_tool(self, sid, tid, name, obs):
        pass

    async def history(self, sid):
        return []

    async def set_summary(self, sid, summary):
        pass


class FakeTools:
    """可调工具：knowledge_search 返回带 [n] 来源的检索结果。"""

    def __init__(self):
        self.calls = []

    def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append(name)
        if name == "knowledge_search":
            return ToolResult(
                success=True,
                output=(
                    "检索来源：用户 test 的知识库「默认」，命中 2 条：\n\n"
                    "[1] 来源:paper_a.txt | 相似度:0.912\n钙钛矿太阳能电池效率已达 25.7%\n\n"
                    "[2] 来源:paper_b.txt | 相似度:0.834\n钙钛矿稳定性瓶颈在于离子迁移"
                ),
            )
        return ToolResult(success=True, output="done")


def _kb_tool_response():
    return LLMResponse(
        content="检索知识库",
        tool_calls=[ToolCall(id="c1", name="knowledge_search", arguments='{"query": "钙钛矿"}')],
        finish_reason="tool_calls",
    )


async def _collect(agent, msg):
    return [e async for e in agent.stream("sid", msg, use_plan=False)]


def test_missing_citation_triggers_rewrite(monkeypatch):
    """用过 knowledge_search 但答案无 [n] → 追加引用要求重写一次。"""
    from app.config import get_settings

    monkeypatch.setattr(get_settings().llm, "aux_model", "")

    llm = FakeLLM([
        _kb_tool_response(),  # 1: 调用工具
        LLMResponse(content="钙钛矿电池效率为 25.7%。", tool_calls=[]),  # 2: 无引用 → 触发重写
        LLMResponse(content="钙钛矿电池效率为 25.7% [1]。稳定性是主要瓶颈 [2]。", tool_calls=[]),  # 3: 带引用
    ])
    tools = FakeTools()
    agent = ResearchAgent(llm=llm, memory=FakeMemory(), tools=tools, planner=None, token_budget=10**9)

    events = asyncio.run(_collect(agent, "介绍一下钙钛矿电池"))
    answers = [e for e in events if e.type == "answer"]
    assert llm.calls >= 3, f"应触发一次重写，实际调用 {llm.calls} 次"
    assert "[1]" in answers[-1].data["content"]
    assert tools.calls.count("knowledge_search") == 1  # 重写不重复检索


def test_with_citation_no_rewrite(monkeypatch):
    """答案已带 [n] → 不触发重写。"""
    from app.config import get_settings
    from app.agent.prompts import CITATION_PROMPT

    monkeypatch.setattr(get_settings().llm, "aux_model", "")
    llm = FakeLLM([
        _kb_tool_response(),
        LLMResponse(content="钙钛矿电池效率为 25.7% [1]。", tool_calls=[]),
    ])
    tools = FakeTools()
    agent = ResearchAgent(llm=llm, memory=FakeMemory(), tools=tools, planner=None, token_budget=10**9)

    events = asyncio.run(_collect(agent, "介绍一下钙钛矿电池"))
    answers = [e for e in events if e.type == "answer"]
    # 2 次主流程调用（工具 + 带引用答案）+ 1 次后台答案质检 = 3
    assert llm.calls == 3, f"不应重写，实际调用 {llm.calls} 次"
    assert "[1]" in answers[-1].data["content"]
    # 引用规范提示不应被注入
    assert not any(CITATION_PROMPT[:50] in m for m in llm.seen_system_messages)


def test_no_retrieval_no_citation_requirement(monkeypatch):
    """未使用检索工具 → 不强制引用，正常回答。"""
    from app.config import get_settings

    monkeypatch.setattr(get_settings().llm, "aux_model", "")
    llm = FakeLLM([
        LLMResponse(content="1+1=2。", tool_calls=[]),
    ])
    tools = FakeTools()
    agent = ResearchAgent(llm=llm, memory=FakeMemory(), tools=tools, planner=None, token_budget=10**9)

    events = asyncio.run(_collect(agent, "1+1 等于几"))
    answers = [e for e in events if e.type == "answer"]
    assert llm.calls == 1
    assert "1+1=2" in answers[-1].data["content"]
