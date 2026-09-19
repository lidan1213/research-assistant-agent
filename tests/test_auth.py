"""登录 / 注册 / 角色权限测试：注册写库、登录、token 校验、admin-student 权限隔离。"""
import uuid

from fastapi.testclient import TestClient

from app.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _login(c: TestClient, username: str, password: str) -> dict:
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    return r


def _register(c: TestClient, username: str, password: str, password2: str | None = None):
    return c.post(
        "/api/auth/register",
        json={"username": username, "password": password, "password2": password2 or password},
    )


def test_register_success_and_login():
    """注册成功：返回 token 且角色为 student，随后可用新账号登录。"""
    c = _client()
    name = f"newbie_{uuid.uuid4().hex[:8]}"
    r = _register(c, name, "pass1234")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["role"] == "student"
    assert j["username"] == name
    assert j["token"]

    # 新账号可直接登录（密码正确）
    r2 = _login(c, name, "pass1234")
    assert r2.status_code == 200


def test_register_duplicate_rejected():
    """重复注册同一账号被拒绝。"""
    c = _client()
    name = f"dup_{uuid.uuid4().hex[:8]}"
    assert _register(c, name, "pass1234").status_code == 200
    r = _register(c, name, "pass1234")
    assert r.status_code == 401
    assert "已存在" in r.json()["message"]


def test_register_password_mismatch():
    """两次密码不一致被拒绝。"""
    c = _client()
    r = _register(c, f"mismatch_{uuid.uuid4().hex[:8]}", "pass1234", "pass9999")
    assert r.status_code == 401
    assert "不一致" in r.json()["message"]


def test_register_invalid_username():
    """空账号 / 含 __ 的账号被拒绝。"""
    c = _client()
    assert _register(c, "  ", "pass1234").status_code == 401
    assert _register(c, "a__b", "pass1234").status_code == 401


def test_login_admin_and_student():
    c = _client()
    r = _login(c, "admin", "123456")
    assert r.status_code == 200
    assert r.json()["role"] == "admin"
    assert r.json()["token"]

    r = _login(c, "student", "123456")
    assert r.status_code == 200
    assert r.json()["role"] == "student"


def test_login_wrong_password():
    c = _client()
    r = _login(c, "admin", "wrong-pass")
    assert r.status_code == 401


def test_login_unknown_user():
    c = _client()
    r = _login(c, "ghost", "123456")
    assert r.status_code == 401


def test_unauthenticated_blocked():
    c = _client()
    # 未登录访问受保护接口 -> 401
    assert c.get("/api/tools").status_code == 401
    assert c.get("/api/userkb/documents").status_code == 401
    assert c.get("/api/knowledge/health").status_code == 401


def test_invalid_token():
    c = _client()
    r = c.get("/api/tools", headers={"Authorization": "Bearer invalid.token.here"})
    assert r.status_code == 401


def test_me_endpoint():
    c = _client()
    token = _login(c, "admin", "123456").json()["token"]
    r = c.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"username": "admin", "role": "admin"}


def test_student_forbidden_on_evaluation_and_knowledge():
    c = _client()
    token = _login(c, "student", "123456").json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    # 学生可以访问自己的个人知识库
    assert c.get("/api/userkb/documents", headers=h).status_code == 200
    # 知识库健康探活：登录即可访问（前端横幅用）
    assert c.get("/api/knowledge/health", headers=h).status_code == 200
    # 全局知识库管理接口仍 admin 专属
    assert c.post(
        "/api/knowledge/query",
        json={"query": "测试", "top_k": 3},
        headers=h,
    ).status_code == 403
    # 学生可以访问对话相关（tools 列表代表对话链路）
    assert c.get("/api/tools", headers=h).status_code == 200


def test_admin_allowed_on_evaluation_and_knowledge():
    c = _client()
    token = _login(c, "admin", "123456").json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    assert c.get("/api/userkb/documents", headers=h).status_code == 200
    assert c.get("/api/knowledge/health", headers=h).status_code == 200
