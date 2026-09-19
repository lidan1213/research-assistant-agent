"""Golden Set 评测框架单测：验证断言匹配、类别过滤、失败检测逻辑。

真实评测（需真实 LLM）跑：PYTHONPATH= .venv/Scripts/python.exe scripts/eval_golden.py
"""
from __future__ import annotations

import asyncio

from scripts.eval_golden import GOLDEN_CASES


def test_golden_cases_structure():
    """用例结构完整：question / category / must_contain 均存在。"""
    assert len(GOLDEN_CASES) >= 3
    for case in GOLDEN_CASES:
        assert case["question"].strip()
        assert case["category"] in ("calculator", "kb", "search", "plain")
        assert isinstance(case["must_contain"], list) and case["must_contain"]
        assert all(isinstance(k, str) and k for k in case["must_contain"])


def test_golden_cases_cover_core_behaviors():
    """用例覆盖三类核心行为：计算/知识库/普通问答。"""
    cats = {c["category"] for c in GOLDEN_CASES}
    assert {"calculator", "kb", "plain"} <= cats


def test_assertion_logic():
    """断言匹配逻辑：答案含所有 must_contain 才通过。"""
    answer = "量子点尺寸通常在 2-10 纳米范围"
    assert all(k in answer for k in ["量子点", "纳米"])
    assert not all(k in answer for k in ["量子点", "带隙"])  # 缺"带隙"应失败


def test_category_filter():
    """类别过滤：只取指定类别的用例。"""
    calc = [c for c in GOLDEN_CASES if c["category"] == "calculator"]
    assert calc and all(c["category"] == "calculator" for c in calc)


def test_run_case_answer_collection():
    """_run_case 收集回答（用 stub agent 验证流程不抛异常）。"""
    from app.agent.agent import ResearchAgent
    from app.llm.base import LLMResponse
    from app.tools.base import ToolRegistry

    class _StubLLM:
        async def chat(self, *a, **k):
            return LLMResponse(content="结果是 391，这是 23×17 的计算结果。")

        async def chat_stream(self, *a, **k):
            from app.llm.base import StreamChunk

            resp = await self.chat(*a, **k)
            yield StreamChunk(content_delta="", tool_calls=[], finish_reason="stop")

    import uuid

    from app.agent.memory import ConversationMemory, SQLiteStore

    store = SQLiteStore(f"var/golden-{uuid.uuid4().hex}.db")
    mem = ConversationMemory(backend=store)
    agent = ResearchAgent(llm=_StubLLM(), memory=mem, tools=ToolRegistry(), planner=None)
    # 直接调用 agent.run 验证回答包含断言
    answer = asyncio.run(agent.run("golden__calc_1", "23*17=?", use_plan=False))
    assert "391" in answer
    asyncio.run(mem.clear("golden__calc_1"))
    store.close()
