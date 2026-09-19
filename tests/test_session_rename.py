"""会话标题重命名测试：改名生效 / 越权拒绝 / 空标题拒绝。"""
from __future__ import annotations

import asyncio
import uuid

from app.agent.memory import SQLiteStore
from app.config import get_settings
from app.llm.base import ChatMessage, MessageRole


def _mk_session(username: str) -> str:
    """创建一条测试会话，返回短 session_id（无前缀）。"""
    sid = f"{username}__t_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        asyncio.run(store.append(sid, ChatMessage(role=MessageRole.USER, content="测试消息")))
    finally:
        store.close()
    return sid.split("__", 1)[1]


def test_session_rename_success(client):
    """重命名自己的会话成功，标题更新。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = client.post(f"/api/chat/sessions/{short}/rename", headers=h, json={"title": "量子点研究"})
    assert r.status_code == 200
    assert r.json()["title"] == "量子点研究"
    # 会话列表应显示新标题
    lst = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {token}"}).json()["sessions"]
    entry = next((s for s in lst if s["session_id"] == short), None)
    assert entry is not None and entry["title"] == "量子点研究"
    # 清理
    _cleanup(f"student1__{short}")


def test_session_rename_cross_user_forbidden(client):
    """student2 不能重命名 student1 的会话。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = client.post(f"/api/chat/sessions/{short}/rename", headers=h, json={"title": "篡改"})
    assert r.status_code == 403
    _cleanup(f"student1__{short}")


def test_session_rename_empty_title_rejected(client):
    """空标题被拒绝。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = client.post(f"/api/chat/sessions/{short}/rename", headers=h, json={"title": "   "})
    assert r.status_code == 403
    _cleanup(f"student1__{short}")


def _cleanup(full_sid: str) -> None:
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        asyncio.run(store.clear(full_sid))
    finally:
        store.close()
