"""会话分享测试：生成链接 / 公开只读查看 / 签名无效拒绝 / 越权拒绝。"""
from __future__ import annotations

import asyncio
import uuid

from app.agent.memory import SQLiteStore
from app.config import get_settings
from app.llm.base import ChatMessage, MessageRole


def _mk_session(username: str) -> str:
    sid = f"{username}__share_{uuid.uuid4().hex[:8]}"
    store = SQLiteStore(get_settings().memory.sqlite_path)
    try:
        store.set_title(sid, "分享测试")
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


def test_share_generate_and_view(client):
    """生成分享链接，且未登录也能公开查看（只读 HTML）。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student1", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = client.post(f"/api/chat/sessions/{short}/share", headers=h)
    assert r.status_code == 200
    j = r.json()
    assert j["url"].startswith("/api/chat/share/")
    assert j["expires_in"] > 0

    # 未登录访问分享链接（无 Authorization header）
    r2 = client.get(j["url"])
    assert r2.status_code == 200
    assert "text/html" in r2.headers.get("content-type", "")
    assert "分享测试" in r2.text
    assert "量子点电池效率" in r2.text  # 用户消息在只读视图中
    _cleanup(f"student1__{short}")


def test_share_cross_user_forbidden(client):
    """student2 不能为 student1 的会话生成分享链接。"""
    short = _mk_session("student1")
    token = client.post("/api/auth/login", json={"username": "student2", "password": "123456"}).json()["token"]
    r = client.post(f"/api/chat/sessions/{short}/share", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    _cleanup(f"student1__{short}")


def test_share_invalid_signature_rejected(client):
    """篡改签名的分享链接被拒绝。"""
    r = client.get("/api/chat/share/abc.def.ghij")
    assert r.status_code == 403
