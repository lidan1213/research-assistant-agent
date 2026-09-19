"""应用与路由冒烟测试。"""
from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_tools():
    client = TestClient(create_app())
    # /api/tools 需要登录：先登录拿 token
    r = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert r.status_code == 200
    client.headers.update({"Authorization": f"Bearer {r.json()['token']}"})
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert {"calculator", "web_search", "arxiv_search", "pdf_reader", "code_executor", "citation"} <= names
