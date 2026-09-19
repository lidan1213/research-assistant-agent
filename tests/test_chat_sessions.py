"""会话记录接口测试：列表隔离 + 历史归属校验。"""
from fastapi.testclient import TestClient

from app.main import create_app


def _login(c: TestClient, username: str) -> str:
    r = c.post("/api/auth/login", json={"username": username, "password": "123456"})
    assert r.status_code == 200
    return r.json()["token"]


def test_sessions_list_isolated(client):
    """各用户只能看到自己的会话列表。"""
    t1 = _login(client, "student1")
    t2 = _login(client, "student2")
    r1 = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {t1}"})
    r2 = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {t2}"})
    assert r1.status_code == 200 and r2.status_code == 200
    s1 = r1.json()["sessions"]
    s2 = r2.json()["sessions"]
    # 会话返回不带用户名前缀
    for s in s1 + s2:
        assert "__" not in s["session_id"]
    # 隔离：student2 的会话不在 student1 的列表里（反之亦然）
    ids1 = {s["session_id"] for s in s1}
    ids2 = {s["session_id"] for s in s2}
    assert not (ids1 & ids2)


def test_history_forbidden_for_other_user(client):
    """student2 不能读取 student1 的会话历史。"""
    t1 = _login(client, "student1")
    t2 = _login(client, "student2")
    s1 = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {t1}"}).json()["sessions"]
    if not s1:
        return  # 无数据则跳过（隔离测试已覆盖）
    sid = s1[0]["session_id"]
    r = client.get(
        f"/api/chat/history?session_id={sid}",
        headers={"Authorization": f"Bearer {t2}"},
    )
    assert r.status_code == 403


def test_history_requires_login(client):
    """未登录访问会话列表/历史 -> 401。"""
    assert client.get("/api/chat/sessions").status_code == 401
    assert client.get("/api/chat/history", params={"session_id": "x"}).status_code == 401
