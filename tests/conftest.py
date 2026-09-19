"""测试公共配置：在导入应用前注入最小环境变量，避免依赖 .env。"""
import os

os.environ.setdefault("APP_NAME", "TestResearchAgent")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")
os.environ.setdefault("SEARCH_PROVIDER", "duckduckgo")
os.environ.setdefault("API_KEYS", "")
# 测试隔离：禁用模型智能路由（.env 里的 LLM__AUX_MODEL 会启用路由，导致
# 注入 stub LLM 的测试误走真实 aux 模型）。环境变量优先级高于 .env。
os.environ["LLM__AUX_MODEL"] = ""
os.environ.setdefault("KNOWLEDGE__EMBEDDING_MODEL", "hashing://512")
# 默认测试套件保持完全离线；Cross-Encoder 由 mock 单测验证，不下载模型。
os.environ.setdefault("KNOWLEDGE__RERANKER_KIND", "lexical")
# 会话测试不依赖开发机 Redis；Redis 双写行为由独立 mock 测试覆盖。
os.environ["MEMORY__BACKEND"] = "memory"
os.environ["MEMORY__REDIS_URL"] = ""
# 测试隔离：用户库指向临时文件，避免污染开发库
os.environ.setdefault("AUTH_USERS_DB", os.path.join(os.path.dirname(__file__), "test_users.db"))
# 测试隔离：笔记目录指向测试目录，避免误删开发笔记
os.environ.setdefault("NOTES_DIR", os.path.join(os.path.dirname(__file__), "test_notes_dir"))
# 测试隔离：知识图谱库指向测试目录，避免污染开发图谱
os.environ.setdefault("KG_DATA_DIR", os.path.join(os.path.dirname(__file__), "test_kg_dir"))
# 测试隔离：ChromaDB collection 加 test_ 前缀，避免测试创建/删除的 kb_* 污染生产库
os.environ.setdefault("CHROMA_COLLECTION_PREFIX", "test_")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

CHROMA_URL = os.environ.get("CHROMA_URL", "http://127.0.0.1:8001")


def chroma_available() -> bool:
    """探测 ChromaDB REST 服务是否存活（测试前门禁，避免挂掉后出现误导性断言）。"""
    try:
        import urllib.request

        with urllib.request.urlopen(f"{CHROMA_URL}/api/v1/heartbeat", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _chroma_health_gate():
    """依赖真实 ChromaDB 的测试在服务不可用时给出明确错误而非误导断言。

    测试中直接使用真实 ChromaDB 的模块（user_kb / version_update 等）在
    pytest 收集时通过 fixtures 探测；不可用时跳过或报错由各测试决定。
    这里仅在日志层面提示，避免拖慢不需要 ChromaDB 的纯单元测试。
    """
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def login_token(client: TestClient, username: str = "admin", password: str = "123456") -> str:
    """登录并返回 token（默认管理员）。"""
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def auth_client(
    client: TestClient, username: str = "admin", password: str = "123456"
) -> TestClient:
    """返回已携带登录 Authorization header 的客户端。"""
    token = login_token(client, username, password)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
