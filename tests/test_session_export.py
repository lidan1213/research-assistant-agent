"""会话导出测试：Markdown 导出成功 / 越权拒绝 / 会话不存在拒绝。"""
from __future__ import annotations

import asyncio
import uuid

from app.agent.memory import SQLiteStore
from app.config import get_settings
from app.llm.base import ChatMessage, MessageRole


def _mk_session(username: str) -> str:
    """创建一条带标题/摘要/消息的测试会话，返回短 session_id。"""
    sid = f"{username}__exp_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.set_title(sid, "导出测试")
        store.set_summary(sid, "摘要内容")
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.USER, content="你好")))
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.ASSISTANT, content="你好！有什么可以帮你？")))
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


def test_session_export_markdown(client):
    """导出自己的会话：返回 Markdown，含标题/摘要/对话内容。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = client.get(f"/api/chat/sessions/{short}/export", headers=h)
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    assert "# 导出测试" in r.text
    assert "摘要内容" in r.text
    assert "你好" in r.text
    _cleanup(f"student1__{short}")


def test_session_export_cross_user_forbidden(client):
    """student2 不能导出 student1 的会话。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    r = client.get(f"/api/chat/sessions/{short}/export", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    _cleanup(f"student1__{short}")


def test_session_export_not_found(client):
    """导出不存在的会话：403（归属校验拦截）。"""
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    r = client.get("/api/chat/sessions/ghost_session/export", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
