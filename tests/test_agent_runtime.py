"""Agent Runtime 拆分测试：guards 纯函数 + ContextBuilder 上下文构建。"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.context import ContextBuilder
from app.agent.guards import (
    answer_needs_revision,
    estimate_tokens,
    is_simple_task,
    over_budget,
    recommended_tool,
    tool_matches_recommendation,
)
from app.llm.base import ChatMessage, MessageRole


# ---------- Guards ----------
def test_is_simple_task_basic():
    assert is_simple_task("你好") is True
    assert is_simple_task("谢谢") is True
    # 带图片 → 复杂
    assert is_simple_task("这是什么", images=["data:image/png;base64,xx"]) is False
    # 超长 → 复杂
    assert is_simple_task("长" * 61) is False
    # 复杂关键词 → 复杂
    assert is_simple_task("帮我总结一下") is False
    assert is_simple_task("计算 1+1") is False
    assert is_simple_task("钙钛矿效率多少") is False
    # 空消息 → 复杂（安全兜底）
    assert is_simple_task("") is False


@pytest.mark.parametrize(
    "query",
    [
        "RAG 的原理是什么？",
        "解释 ReAct 模式的 Thought、Action 和 Observation",
        "Self-RAG 的自我反思机制有什么作用？",
        "大语言模型驱动的 Agent 通常具备哪些核心能力？",
        "Llama 2 是什么？",
    ],
)
def test_knowledge_questions_do_not_bypass_tool_routing(query: str):
    """短知识问题也必须进入完整工具链，不能被简单任务路由直接回答。"""
    assert recommended_tool(query) == "knowledge_search"
    assert is_simple_task(query) is False


def test_estimate_tokens_and_budget():
    msgs = [
        ChatMessage(role=MessageRole.USER, content="你好世界"),
        ChatMessage(role=MessageRole.ASSISTANT, content="这是一个较长的回答内容" * 10),
    ]
    tokens = estimate_tokens(msgs)
    assert tokens > 0
    assert over_budget(msgs, token_budget=1) is True
    assert over_budget(msgs, token_budget=10**9) is False
    assert over_budget(msgs, token_budget=0) is False  # 0 = 不限制


def test_deterministic_tool_routing_and_answer_validation():
    assert recommended_tool("计算 23 乘以 17") == "calculator"
    assert recommended_tool("用 Python 统计字符") == "code_executor"
    assert recommended_tool("在 arXiv 检索 RAG 论文") == "arxiv_search"
    assert recommended_tool("请解释 RRF 的作用") == "knowledge_search"
    assert recommended_tool("你好") == ""
    assert tool_matches_recommendation("knowledge_search", ["kg_query"])
    assert not tool_matches_recommendation("calculator", ["knowledge_search"])
    assert answer_needs_revision("（本次未能生成有效回答，请换一种问法重试）")
    assert not answer_needs_revision("391")


# ---------- ContextBuilder ----------
class StubMemory:
    def __init__(self, history: list[ChatMessage] | None = None):
        self._history = history or []

    async def history(self, session_id: str) -> list[ChatMessage]:
        return list(self._history)


class StubLongTerm:
    def __init__(self, facts: list[dict] | None = None):
        self._facts = facts or []

    def search_facts(self, query: str, k: int = 3) -> list[dict]:
        return self._facts[:k]


def test_context_builder_basic():
    mem = StubMemory(
        [ChatMessage(role=MessageRole.USER, content="hi"), ChatMessage(role=MessageRole.ASSISTANT, content="hello")]
    )
    builder = ContextBuilder(mem, token_budget=0, keep_msgs=16)
    msgs, consumed = asyncio.run(builder.build("s1", plan_steps=["步骤1", "步骤2"]))
    assert msgs[0].role == MessageRole.SYSTEM
    assert "步骤1" in msgs[0].content  # 计划注入
    assert len(msgs) == 3  # system + 2 历史
    assert consumed == []


def test_context_builder_longterm_injection():
    mem = StubMemory([ChatMessage(role=MessageRole.USER, content="我在研究钙钛矿")])
    lt = StubLongTerm([{"content": "用户研究方向是钙钛矿电池", "category": "user_fact"}])
    builder = ContextBuilder(mem, longterm=lt, token_budget=0)
    msgs, _ = asyncio.run(builder.build("alice__s1"))
    assert "钙钛矿电池" in msgs[0].content  # 长期记忆注入 system


def test_context_builder_multimodal_attach():
    mem = StubMemory([ChatMessage(role=MessageRole.USER, content="看这张图")])
    builder = ContextBuilder(mem, token_budget=0)
    msgs, consumed = asyncio.run(
        builder.build("s1", pending_images=["data:image/png;base64,AAA"])
    )
    assert consumed == ["data:image/png;base64,AAA"]
    assert msgs[-1].images == ["data:image/png;base64,AAA"]


def test_context_builder_compress_when_over_budget():
    """历史超预算时压缩，压缩边界落在 USER 消息。"""
    long_history = []
    for i in range(10):
        long_history.append(ChatMessage(role=MessageRole.USER, content=f"问题{i}：" + "字" * 200))
        long_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"回答{i}：" + "字" * 300))
    mem = StubMemory(long_history)

    class FakeAuxLLM:
        async def chat(self, messages, **kwargs):
            from app.llm.base import LLMResponse

            return LLMResponse(content="（压缩摘要）")

    builder = ContextBuilder(mem, token_budget=500, keep_msgs=4)
    # stub get_aux_llm（普通函数返回实例，勿用 async def）
    import app.llm.factory as factory_mod

    def fake_get_aux_llm():
        return FakeAuxLLM()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(factory_mod, "get_aux_llm", fake_get_aux_llm)
    msgs, _ = asyncio.run(builder.build("s1"))
    # 压缩后：msgs[0]=基础 system，msgs[1]=压缩摘要 system，之后是保留的最近消息
    assert msgs[0].role == MessageRole.SYSTEM
    assert "摘要" in msgs[1].content
    # 保留消息数 = 摘要 1 条 + 最近保留（<= keep_msgs 附近）
    assert len(msgs) <= 2 + 4 + 1
    monkeypatch.undo()


def test_agent_delegates_to_guards():
    """ResearchAgent 的 _is_simple_task/_estimate_tokens 委托 guards（行为等价）。"""
    from app.agent.agent import ResearchAgent

    assert ResearchAgent._is_simple_task("你好", None) is True
    assert ResearchAgent._is_simple_task("总结一下", None) is False
