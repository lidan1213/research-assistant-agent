"""Trace（链路追踪）/ PromptGuard（注入防护）/ ModelRouter（多模型路由）测试。"""
from __future__ import annotations

import os
import tempfile

import pytest

from app.agent.agent import ResearchAgent
from app.llm.base import LLMResponse
from app.tools.base import ToolResult


# ---------------- Trace ----------------

def test_trace_ledger_record_and_query(tmp_path):
    """Trace 账本：写入 LLM/工具记录，可按会话查询。"""
    from app.llm.trace import TraceLedger

    ledger = TraceLedger(str(tmp_path / "traces.db"))
    ledger.record("llm", "deepseek-chat", session_id="s1", success=True, duration_ms=120, detail="prompt=10 completion=5")
    ledger.record("tool", "calculator", session_id="s1", success=True, duration_ms=5)
    ledger.record("tool", "web_search", session_id="s1", success=False, duration_ms=3000, detail="timeout")
    ledger.record("llm", "deepseek-chat", session_id="s2", success=True, duration_ms=80)

    rows = ledger.query("s1")
    assert len(rows) == 3
    assert rows[0]["kind"] == "tool"  # 按 id 倒序
    assert rows[0]["success"] is False

    summary = ledger.session_summary("s1")
    assert summary["llm_calls"] == 1
    assert summary["tool_calls"] == 2
    assert summary["failures"] == 1


def test_trace_ledger_empty_session(tmp_path):
    """未记录过的会话返回空列表，不报错。"""
    from app.llm.trace import TraceLedger

    ledger = TraceLedger(str(tmp_path / "traces.db"))
    assert ledger.query("ghost_session") == []
    assert ledger.session_summary("ghost_session")["llm_calls"] == 0


# ---------------- PromptGuard ----------------

def test_tool_result_wraps_data_marker():
    """PromptGuard：工具成功输出统一包裹 <tool_data> 标记。"""
    r = ToolResult(success=True, output={"results": [1, 2]})
    content = r.to_message_content()
    assert content.startswith("<tool_data>") and content.endswith("</tool_data>")
    assert '"results"' in content

    # 失败结果不包裹（保持错误可读）
    err = ToolResult(success=False, error="boom")
    assert not err.to_message_content().startswith("<data>")


def test_system_prompt_contains_guard():
    """PromptGuard：system prompt 含外部数据不可信声明。"""
    from app.agent.prompts import SYSTEM_PROMPT

    assert "未经信任的外部数据" in SYSTEM_PROMPT
    assert "PromptGuard" in SYSTEM_PROMPT


# ---------------- ModelRouter ----------------

class _StubLLM:
    def __init__(self, name="main"):
        self.name = name
        self.calls = 0

    async def chat(self, *a, **k):
        self.calls += 1
        return LLMResponse(content="ok")


def test_aux_llm_fallback_to_main_without_config(monkeypatch):
    """未配置 aux_model 时，_aux_llm 回退主模型（stub 可测）。"""
    from app.config import get_settings

    monkeypatch.setattr(get_settings().llm, "aux_model", "")
    main_llm = _StubLLM("main")
    agent = ResearchAgent(llm=main_llm, memory=object(), tools=None)
    # 替换 memory 为可用的假对象（只测 _aux_llm，不触发其他逻辑）
    assert agent._aux_llm() is main_llm


def test_aux_llm_uses_aux_when_configured(monkeypatch):
    """配置 aux_model 时，_aux_llm 返回辅助模型实例（与主模型不同）。"""
    from app.config import get_settings
    from app.llm import factory as factory_mod

    monkeypatch.setattr(get_settings().llm, "aux_model", "fake-aux-model")
    # 缓存清除，确保新实例
    factory_mod._cache.clear()
    try:
        main_llm = _StubLLM("main")
        agent = ResearchAgent(llm=main_llm, memory=object(), tools=None)
        aux = agent._aux_llm()
        assert aux is not main_llm
        assert aux.model == "fake-aux-model"
    finally:
        factory_mod._cache.clear()
        monkeypatch.setattr(get_settings().llm, "aux_model", "")


def test_factory_get_llm_model_override(monkeypatch):
    """factory.get_llm 支持 model 覆盖且按完整键缓存。"""
    from app.config import get_settings
    from app.llm import factory as factory_mod

    monkeypatch.setattr(get_settings().llm, "model", "main-model")
    factory_mod._cache.clear()
    try:
        a = factory_mod.get_llm()
        b = factory_mod.get_llm()
        assert a is b  # 同参缓存复用
        c = factory_mod.get_llm(model="other-model")
        assert c is not a
        assert c.model == "other-model"
    finally:
        factory_mod._cache.clear()
