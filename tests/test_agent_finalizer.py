"""SessionFinalizer behavior without external services."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent.finalizer import SessionFinalizer
from app.llm.base import ChatMessage, MessageRole


class _Memory:
    def __init__(self):
        self.summary = ""

    async def history(self, session_id):
        return [ChatMessage(role=MessageRole.USER, content="研究钙钛矿电池")]

    async def set_summary(self, session_id, summary):
        self.summary = summary


class _LLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def chat(self, messages, **kwargs):
        return SimpleNamespace(content=next(self.responses))


class _Longterm:
    def __init__(self):
        self.facts = []

    def append_fact(self, session_id, **fact):
        self.facts.append((session_id, fact))


def test_finalizer_summary_evaluation_and_facts():
    memory = _Memory()
    longterm = _Longterm()
    llm = _LLM(
        [
            "会话摘要",
            '{"answered": false, "confidence": "low", "missing": "实验条件"}',
            '[{"content": "效率达到30%", "category": "finding"}]',
        ]
    )
    finalizer = SessionFinalizer(memory, lambda: llm, longterm)
    events = []

    async def callback(session_id, data):
        events.append((session_id, data))

    finalizer.eval_callback = callback

    async def run():
        await finalizer.summarize("s")
        await finalizer.evaluate("s", "实验结果如何？", "效率达到30%，但尚未列出条件。")
        await finalizer.extract_facts("s", "实验结果如何？", "效率达到30%")

    asyncio.run(run())
    assert memory.summary == "会话摘要"
    assert events[0][1]["answered"] is False
    assert longterm.facts[0][1]["content"] == "效率达到30%"


def test_finalizer_ignores_short_answer_evaluation():
    llm = _LLM([])
    finalizer = SessionFinalizer(_Memory(), lambda: llm)
    asyncio.run(finalizer.evaluate("s", "q", "短答"))
