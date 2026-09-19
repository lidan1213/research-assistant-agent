"""Trace 链路测试：trace_id 贯穿记录 + by-trace 查询。"""
from __future__ import annotations

import time

import pytest

from app.llm.trace import TraceLedger, get_trace_ledger, reset_trace_ledger


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """临时 TraceLedger，隔离测试数据。"""
    reset_trace_ledger()
    path = str(tmp_path / "trace_test.db")
    # 直接构造实例（不经过全局单例）
    led = TraceLedger(path)
    yield led


def test_record_and_query_by_trace(ledger: TraceLedger):
    """同 trace_id 的多条记录可按链路查询（正序，trace_id 经 ContextVar 携带）。"""
    from app.llm.trace import current_trace_id

    tid = "trace1234567890"
    tok = current_trace_id.set(tid)
    try:
        # 模拟一次请求链路：llm → tool → llm
        ledger.record("llm", "gpt-5.4", success=True, duration_ms=100, detail="req", session_id="s1")
        ledger.record("tool", "calculator", success=True, duration_ms=5, detail="ok", session_id="s1")
        ledger.record("llm", "gpt-5.4", success=True, duration_ms=200, detail="final", session_id="s1")
    finally:
        current_trace_id.reset(tok)

    chain = ledger.query_by_trace(tid)
    assert len(chain) == 3
    assert [c["kind"] for c in chain] == ["llm", "tool", "llm"]  # 正序
    assert all(c["trace_id"] == tid for c in chain)
    assert chain[0]["duration_ms"] == 100


def test_query_by_trace_isolates_traces(ledger: TraceLedger):
    """不同 trace_id 互不串扰。"""
    from app.llm.trace import current_trace_id

    tok1 = current_trace_id.set("aaa")
    try:
        ledger.record("llm", "m1", session_id="s")
    finally:
        current_trace_id.reset(tok1)
    tok2 = current_trace_id.set("bbb")
    try:
        ledger.record("tool", "t1", session_id="s")
    finally:
        current_trace_id.reset(tok2)
    assert len(ledger.query_by_trace("aaa")) == 1
    assert len(ledger.query_by_trace("bbb")) == 1
    assert ledger.query_by_trace("nonexistent") == []


def test_record_empty_trace_id(ledger: TraceLedger):
    """无 trace_id 时也记录（trace_id 为空串），不影响查询。"""
    ledger.record("llm", "m1", session_id="s")
    rows = ledger.query(session_id="s")
    assert len(rows) == 1
    assert rows[0]["trace_id"] == ""
    assert ledger.query_by_trace("") == []  # 空 trace 不按链路查


def test_global_ledger_contextvar(monkeypatch, tmp_path):
    """get_trace_ledger 单例 + current_trace_id ContextVar 自动携带。"""
    from app.llm.trace import current_trace_id

    reset_trace_ledger()
    led = get_trace_ledger()
    # 替换为临时 DB（避免污染开发库）
    led_path = str(tmp_path / "trace_global.db")
    import app.llm.trace as trace_mod

    monkeypatch.setattr(trace_mod, "_db_path", lambda: led_path)
    reset_trace_ledger()
    led = get_trace_ledger()

    tok = current_trace_id.set("ctx_trace_1")
    try:
        led.record("llm", "gpt-5.4", success=True, duration_ms=10)
    finally:
        current_trace_id.reset(tok)
    chain = led.query_by_trace("ctx_trace_1")
    assert len(chain) == 1
    assert chain[0]["name"] == "gpt-5.4"
    reset_trace_ledger()
