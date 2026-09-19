"""安全加固与审计测试：注册密码强度 / 登录限流 / 审计接口 / 评测接口。"""
from __future__ import annotations

import time as _time

from app.core.audit import check_sensitive, record_audit
from app.llm.trace import get_trace_ledger, reset_trace_ledger


def _client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def _register(c, username, password, password2=None):
    return c.post("/api/auth/register", json={"username": username, "password": password, "password2": password2 or password})


def test_register_password_strength():
    """注册密码强度：≥8 位且含字母数字。"""
    c = _client()
    import uuid

    # 太短
    r = _register(c, f"w1_{uuid.uuid4().hex[:6]}", "short")
    assert r.status_code == 401
    assert "8 位" in r.json()["message"]
    # 纯字母（无数字）
    r = _register(c, f"w2_{uuid.uuid4().hex[:6]}", "abcdefgh")
    assert r.status_code == 401
    # 合法：8 位含字母数字
    r = _register(c, f"w3_{uuid.uuid4().hex[:6]}", "pass1234")
    assert r.status_code == 200


def test_login_rate_limit():
    """连续 5 次失败后锁定，第 6 次提示锁定。"""
    from app.api.routes import auth as auth_mod

    with auth_mod._login_lock:
        auth_mod._login_fails.clear()
    c = _client()
    for _ in range(5):
        r = c.post("/api/auth/login", json={"username": "rate_limited_user", "password": "wrong"})
        assert r.status_code == 401
    # 第 6 次：锁定提示（即使密码正确）
    r = c.post("/api/auth/login", json={"username": "rate_limited_user", "password": "123456"})
    assert r.status_code == 401
    assert "锁定" in r.json()["message"]


def test_check_sensitive_words():
    assert check_sensitive("这篇论文讨论毒品成瘾机制") == ["毒品"]
    assert check_sensitive("量子点电池研究") == []


def test_audit_endpoint(tmp_path):
    """审计接口：敏感词记录可被管理端查询。"""
    import os

    os.environ["TRACE_DB"] = str(tmp_path / "traces_audit.db")
    reset_trace_ledger()
    record_audit("student1", "sensitive", {"word": "作弊", "message": "帮我作弊", "session": "s1"})
    record_audit("student1", "unauthorized", {"kind": "越权访问他人会话", "target": "admin__x"})

    c = _client()
    token = c.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = c.get("/api/admin/audit?days=7", headers=h)
    assert r.status_code == 200
    j = r.json()
    assert j["total"] >= 2
    assert j["sensitive_total"] >= 1
    assert any(w["word"] == "作弊" for w in j["top_words"])

    del os.environ["TRACE_DB"]
    reset_trace_ledger()


def test_evaluate_endpoint(monkeypatch, tmp_path):
    """评测接口：POST 启动任务返回 task_id，GET 轮询返回 done（stub 报告）。"""
    import asyncio
    import os

    async def _fake_run_all(db_path, category=None):
        await asyncio.sleep(0)
        return {
            "total": 2, "passed": 2, "failed": 0, "pass_rate": 1.0,
            "results": [
                {"category": "calculator", "question": "23*17", "must_contain": ["391"], "ok": True, "detail": "391"},
                {"category": "plain", "question": "RAG", "must_contain": ["检索"], "ok": True, "detail": "ok"},
            ],
        }

    monkeypatch.setattr("app.eval_golden.run_all", _fake_run_all)

    c = _client()
    token = c.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = c.post("/api/admin/evaluate", headers=h)
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    # 轮询直到 done（stub 立即完成）
    import time

    for _ in range(20):
        j = c.get(f"/api/admin/evaluate/{task_id}", headers=h).json()
        if j["status"] == "done":
            break
        time.sleep(0.2)
    assert j["status"] == "done"
    assert j["report"]["passed"] == 2


def test_answer_evaluation_api():
    """答案级评测接口在不调用外部 LLM 时可离线返回确定性指标。"""
    c = _client()
    token = c.post(
        "/api/auth/login", json={"username": "admin", "password": "123456"}
    ).json()["token"]
    response = c.post(
        "/api/admin/answer-evaluate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Transformer 哪年提出？",
            "answer": "Transformer 于 2017 年提出。",
            "reference_answer": "2017 年",
            "contexts": ["Attention Is All You Need 论文发表于 2017 年。"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["answer_correctness"] <= 1
    assert 0 <= body["groundedness"] <= 1
    assert body["method"] == "deterministic"
