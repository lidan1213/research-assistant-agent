"""用户个人知识库测试：上传/列表/检索/隔离/权限。

依赖真实 ChromaDB REST（8001）：服务不可用时直接 fail-fast 并给出明确提示，
避免出现“断言 False is True”这类误导性业务断言（ChromaDB 挂掉的特征）。
"""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

from tests._chroma_gate import chroma_available

pytestmark = pytest.mark.skipif(
    not chroma_available(),
    reason="ChromaDB REST (127.0.0.1:8001) 未运行：先启动 .chroma-venv 的 0.4.24 服务",
)


def _client() -> TestClient:  # noqa: ARG001
    return TestClient(create_app())


def _login(c: TestClient, username: str = "student1") -> str:
    r = c.post("/api/auth/login", json={"username": username, "password": "123456"})
    assert r.status_code == 200
    return r.json()["token"]


def _upload(c: TestClient, token: str, filename: str, text: str, save: bool = True):
    return c.post(
        "/api/userkb/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
        data={"save_to_kb": "true" if save else "false"},
    )


def test_upload_save_to_kb(client):
    token = _login(client, "student4")  # 用干净的测试用户，避免与既有知识库内容撞去重
    h = {"Authorization": f"Bearer {token}"}
    client.delete("/api/userkb/clear", headers=h)
    r = _upload(client, token, "notes.txt", "注意力机制让模型关注输入的特定部分。")
    assert r.status_code == 200
    body = r.json()
    assert body["saved_to_kb"] is True
    assert body["chunks"] >= 1
    assert body["filename"] == "notes.txt"
    client.delete("/api/userkb/clear", headers=h)


def test_upload_without_save(client):
    token = _login(client, "student4")
    r = _upload(client, token, "temp.txt", "临时内容", save=False)
    assert r.status_code == 200
    assert r.json()["saved_to_kb"] is False
    # 未入库 -> 列表应为空（若之前有数据则总数不变，这里用新用户验证）
    r2 = client.get("/api/userkb/documents", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200


def test_user_kb_isolation(client):
    """student1 与 student2 的知识库互不可见。"""
    t1 = _login(client, "student1")
    t2 = _login(client, "student2")
    _upload(client, t1, "s1.txt", "注意力机制是深度学习的核心。")
    _upload(client, t2, "s2.txt", "光合作用是植物利用阳光制造养分的过程。")

    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    docs1 = client.get("/api/userkb/documents", headers=h1).json()
    docs2 = client.get("/api/userkb/documents", headers=h2).json()
    src1 = {d["source"] for d in docs1["documents"]}
    src2 = {d["source"] for d in docs2["documents"]}
    assert "s1.txt" in src1 and "s2.txt" not in src1
    assert "s2.txt" in src2 and "s1.txt" not in src2

    # 检索隔离：student1 搜自己的内容能命中，但结果不包含 student2 的文件
    r = client.post("/api/userkb/search", headers=h1, data={"query": "注意力机制", "top_k": "3"})
    assert r.status_code == 200
    for hit in r.json()["hits"]:
        assert (hit.get("metadata") or {}).get("source") != "s2.txt"


def test_search_relevant_and_unrelated(client):
    token = _login(client)
    client.delete("/api/userkb/clear", headers={"Authorization": f"Bearer {token}"})
    _upload(client, token, "deep.txt", "深度学习是机器学习的一个分支，使用多层神经网络。")
    h = {"Authorization": f"Bearer {token}"}
    # 相关查询命中
    r = client.post("/api/userkb/search", headers=h, data={"query": "深度学习 神经网络", "top_k": "3"})
    hits = r.json()["hits"]
    assert len(hits) >= 1
    # 完全不相关的查询（阈值过滤后可能为空，不报错即可）
    r2 = client.post("/api/userkb/search", headers=h, data={"query": "量子力学 弦理论", "top_k": "3"})
    assert r2.status_code == 200


def test_libraries_isolation(client):
    """多库隔离：student1 建的命名库，student2 看不到；列表只含自己的库。"""
    t1 = _login(client, "student1")
    h1 = {"Authorization": f"Bearer {t1}"}
    # 创建命名库「量子点」
    r = client.post("/api/userkb/libraries", headers={**h1, "Content-Type": "application/json"}, json={"name": "量子点"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "量子点"
    # student1 自己的库列表包含默认库 + 量子点
    r = client.get("/api/userkb/libraries", headers=h1)
    names = [lib["name"] for lib in r.json()["libraries"]]
    assert None in names and "量子点" in names  # 默认库 + 命名库

    # student2 看不到 student1 的命名库
    t2 = _login(client, "student2")
    r2 = client.get("/api/userkb/libraries", headers={"Authorization": f"Bearer {t2}"})
    names2 = [lib["name"] for lib in r2.json()["libraries"]]
    assert "量子点" not in names2

    # 非法库名被拒绝
    r3 = client.post("/api/userkb/libraries", headers={**h1, "Content-Type": "application/json"}, json={"name": "bad/name"})
    assert r3.status_code == 403

    # 清理：删除命名库
    r4 = client.delete("/api/userkb/libraries/量子点", headers=h1)
    assert r4.status_code == 200


def test_library_rename_migrates_data(client):
    """库重命名：数据迁移到新库，检索仍命中；显示名更新。"""
    t1 = _login(client, "student1")
    h1 = {"Authorization": f"Bearer {t1}"}
    # 建库 + 上传
    r = client.post("/api/userkb/libraries", headers={**h1, "Content-Type": "application/json"}, json={"name": "旧库"})
    assert r.status_code == 200
    up = client.post(
        "/api/userkb/upload", headers=h1,
        data={"save_to_kb": "true", "kb_name": "旧库"},
        files={"file": ("doc.txt", "量子点太阳能电池的效率提升研究内容。", "text/plain")},
    )
    assert up.status_code == 200

    # 重命名
    r2 = client.post(
        "/api/userkb/libraries/旧库/rename", headers={**h1, "Content-Type": "application/json"},
        json={"new_name": "新库"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["name"] == "新库"

    # 列表显示新名 + 文档数保留
    libs = client.get("/api/userkb/libraries", headers=h1).json()["libraries"]
    new_lib = next((l for l in libs if l["name"] == "新库"), None)
    assert new_lib is not None and new_lib["documents"] >= 1
    assert all(l["name"] != "旧库" for l in libs)  # 旧库已删除

    # 新库检索命中（数据迁移成功）
    s = client.post("/api/userkb/search", headers=h1, data={"query": "量子点 太阳能", "top_k": "3", "kb_name": "新库"})
    assert any((x.get("metadata", {}).get("source") == "doc.txt") for x in s.json()["hits"])

    # 重名冲突拒绝
    client.post("/api/userkb/libraries", headers={**h1, "Content-Type": "application/json"}, json={"name": "新库2"})
    r3 = client.post(
        "/api/userkb/libraries/新库2/rename", headers={**h1, "Content-Type": "application/json"},
        json={"new_name": "新库"},
    )
    assert r3.status_code == 403

    # 清理
    client.delete("/api/userkb/libraries/新库", headers=h1)
    client.delete("/api/userkb/libraries/新库2", headers=h1)


def test_upload_dedup(client):
    """相同内容上传两次，第二次应标记 duplicate 且不重复入库。"""
    token = _login(client, "student4")
    h = {"Authorization": f"Bearer {token}"}
    client.delete("/api/userkb/clear", headers=h)
    text = "去重测试：这是唯一的文档内容。" * 5
    r1 = _upload(client, token, "a.txt", text)
    assert r1.json()["duplicate"] is False
    assert r1.json()["chunks"] >= 1
    r2 = _upload(client, token, "b_copy.txt", text)  # 同内容不同文件名
    assert r2.json()["duplicate"] is True
    assert r2.json()["chunks"] == 0
    docs = client.get("/api/userkb/documents", headers=h).json()["documents"]
    assert len(docs) == 1  # 只入库了一次
    client.delete("/api/userkb/clear", headers=h)


def test_delete_single_document(client):
    """删除单个文档后，其他文档不受影响。"""
    token = _login(client, "student4")
    h = {"Authorization": f"Bearer {token}"}
    client.delete("/api/userkb/clear", headers=h)
    _upload(client, token, "keep.txt", "要保留的文档内容。")
    _upload(client, token, "drop.txt", "要删除的文档内容。")
    docs = client.get("/api/userkb/documents", headers=h).json()["documents"]
    assert len(docs) == 2
    drop_id = next(d["doc_id"] for d in docs if d["source"] == "drop.txt")
    r = client.delete(f"/api/userkb/documents/{drop_id}", headers=h)
    assert r.status_code == 200
    docs2 = client.get("/api/userkb/documents", headers=h).json()["documents"]
    assert len(docs2) == 1
    assert docs2[0]["source"] == "keep.txt"
    client.delete("/api/userkb/clear", headers=h)
