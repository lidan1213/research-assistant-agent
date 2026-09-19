"""会话重试 / 会话全文搜索 / 管理端成本趋势 测试。"""
from __future__ import annotations

import asyncio
import os
import uuid

from app.agent.memory import SQLiteStore
from app.config import get_settings
from app.llm.base import ChatMessage, MessageRole


def _mk_session(username: str, tag: str) -> str:
    """创建带用户提问 + 回答的会话，返回短 session_id。"""
    sid = f"{username}__{tag}_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.set_title(sid, f"{tag}测试")
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.USER, content="量子点电池效率？")))
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.ASSISTANT, content="约30%，瓶颈是稳定性。")))
    finally:
        store.close()
    return sid.split("__", 1)[1]


def _cleanup(full_sid: str) -> None:
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        asyncio.run(store.clear(full_sid))
        store._conn.execute("DELETE FROM session_titles WHERE session_id=?", (full_sid,))
        store._conn.commit()
    finally:
        store.close()


# ---------- 重试 ----------


def test_retry_truncates_partial_answer(client):
    """重试：删除最后提问之后的所有半成品消息，保留用户提问。"""
    short = _mk_session("student1", "retry")
    full = f"student1__{short}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        # 追加半成品回答（assistant + 工具消息）
        asyncio.run(store.append(full, ChatMessage(role=MessageRole.ASSISTANT, content="正在检索…")))
        asyncio.run(store.append(full, ChatMessage(role=MessageRole.TOOL, content="tool output", name="search")))
    finally:
        store.close()

    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    r = client.post(f"/api/chat/sessions/{short}/retry", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["removed"] == 3  # 回答 + 半成品 + 工具消息

    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        msgs = asyncio.run(store.get(full))
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER
    finally:
        store.close()
    _cleanup(full)


def test_retry_cross_user_forbidden(client):
    """student2 不能重试 student1 的会话。"""
    short = _mk_session("student1", "retryx")
    token = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    r = client.post(f"/api/chat/sessions/{short}/retry", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    _cleanup(f"student1__{short}")


# ---------- 会话全文搜索 ----------


def test_search_finds_own_sessions_only(client):
    """学生只能搜到自己的会话；admin 可跨学生搜索。"""
    short = _mk_session("student1", "search")
    full = f"student1__{short}"

    t1 = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    t2 = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    ta = client.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()["token"]

    # student1 命中自己的会话
    r1 = client.get("/api/chat/sessions/search?q=量子", headers={"Authorization": f"Bearer {t1}"})
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1["hits"] and j1["hits"][0]["session_id"] == full
    assert "量子点电池效率" in j1["hits"][0]["snippet"]

    # student2 搜不到 student1 的会话
    r2 = client.get("/api/chat/sessions/search?q=量子", headers={"Authorization": f"Bearer {t2}"})
    assert r2.status_code == 200
    assert all(not h["session_id"].startswith("student1__") for h in r2.json()["hits"])

    # admin 跨学生可搜到
    r3 = client.get("/api/chat/sessions/search?q=量子", headers={"Authorization": f"Bearer {ta}"})
    assert r3.status_code == 200
    assert any(h["session_id"] == full for h in r3.json()["hits"])
    _cleanup(full)


def test_search_empty_query_rejected(client):
    """空查询被 422 拒绝。"""
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    r = client.get("/api/chat/sessions/search?q=", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 422


# ---------- 管理端成本趋势 ----------


def test_cost_trend_aggregates(tmp_path, client):
    """成本趋势：按天/按学生聚合 llm trace（token → 估算成本）。"""
    os.environ["TRACE_DB"] = str(tmp_path / "traces_test.db")
    from app.llm.trace import get_trace_ledger, reset_trace_ledger

    reset_trace_ledger()
    ledger = get_trace_ledger()
    ledger.record("llm", "deepseek-chat", session_id="student1__s1", detail="prompt=1000 completion=500")
    ledger.record("llm", "deepseek-chat", session_id="student2__s2", detail="prompt=200 completion=100")

    ta = client.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()["token"]
    r = client.get("/api/admin/cost-trend?days=7", headers={"Authorization": f"Bearer {ta}"})
    assert r.status_code == 200
    j = r.json()
    assert j["daily"], "应有当天聚合数据"
    assert j["daily"][0]["requests"] == 2
    assert j["daily"][0]["tokens"] == 1800
    assert j["daily"][0]["cost"] > 0
    names = {s["username"] for s in j["by_student"]}
    assert {"student1", "student2"} <= names
    # 按成本降序
    costs = [s["cost"] for s in j["by_student"]]
    assert costs == sorted(costs, reverse=True)

    del os.environ["TRACE_DB"]
    reset_trace_ledger()
