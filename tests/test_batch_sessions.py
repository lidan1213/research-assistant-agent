"""会话批量管理测试：批量删除 / 导出全部。"""
from __future__ import annotations

import asyncio
import uuid

from app.agent.memory import SQLiteStore
from app.config import get_settings
from app.llm.base import ChatMessage, MessageRole


def _mk_session(username: str, tag: str) -> str:
    sid = f"{username}__{tag}_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.set_title(sid, f"{tag}批量测试")
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.USER, content=f"{tag}主题问题")))
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.ASSISTANT, content=f"{tag}的结论内容")))
    finally:
        store.close()
    return sid.split("__", 1)[1]


def _cleanup(full_sid: str) -> None:
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.delete_session(full_sid)
    finally:
        store.close()


def _login(client, u):
    return client.post("/api/auth/login", json={"username": u, "password": "123456"}).json()["token"]


def test_batch_delete_own_sessions(client):
    """批量删除自己的会话。"""
    a = _mk_session("student1", "bd1")
    b = _mk_session("student1", "bd2")
    h = {"Authorization": f"Bearer {_login(client, 'student1')}"}
    r = client.post("/api/chat/sessions/batch-delete", headers={**h, "Content-Type": "application/json"}, json={"ids": [a, b]})
    assert r.status_code == 200
    assert r.json()["deleted"] == 2
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        assert not store.list_sessions(prefix="student1__") or all(
            s["session_id"].split("__", 1)[1] not in (a, b) for s in store.list_sessions(prefix="student1__")
        )
    finally:
        store.close()


def test_batch_delete_cross_user_blocked(client):
    """student2 不能批量删除 student1 的会话。"""
    a = _mk_session("student1", "bdx")
    h2 = {"Authorization": f"Bearer {_login(client, 'student2')}"}
    # 用 student2 的 token 请求删除 student1 的会话（id 前缀自动补成 student2__，不属于自己 → 跳过）
    r = client.post("/api/chat/sessions/batch-delete", headers={**h2, "Content-Type": "application/json"}, json={"ids": [a]})
    assert r.status_code == 200
    assert r.json()["deleted"] == 0
    _cleanup(f"student1__{a}")


def test_export_all_contains_sessions(client):
    """导出全部：合并 Markdown 含标题与问答内容。"""
    a = _mk_session("student1", "exp1")
    h = {"Authorization": f"Bearer {_login(client, 'student1')}"}
    r = client.post("/api/chat/sessions/export-all", headers=h)
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    assert "exp1批量测试" in r.text
    assert "exp1主题问题" in r.text
    assert "exp1的结论内容" in r.text
    _cleanup(f"student1__{a}")
