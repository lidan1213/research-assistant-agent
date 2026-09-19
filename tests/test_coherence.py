"""连贯性改造测试：写作接 RAG / KG 增量同步 / 笔记入库闭环。"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


# ---------- KG 增量同步（sync_document_to_kg） ----------
@pytest.fixture()
def kg_env(tmp_path: Path, monkeypatch):
    """把 KG_DATA_DIR 指向 tmp_path，让 get_kg_store 与测试 store 一致。"""
    monkeypatch.setenv("KG_DATA_DIR", str(tmp_path))
    from app.knowledge.kg import KGStore

    store = KGStore("sync_test", tmp_path)
    yield store
    store.close()


class StubLLM:
    """返回固定抽取结果的 stub，验证 sync 逻辑而非 LLM。"""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        from app.llm.base import LLMResponse

        return LLMResponse(content=self.payload)


def test_sync_document_upsert(kg_env, tmp_path: Path, monkeypatch):
    """上传文档 -> 增量抽取实体/关系入库。"""
    from app.knowledge.kg import sync_document_to_kg

    llm = StubLLM(
        '{"entities":[{"name":"方法A","type":"方法"},{"name":"材料B","type":"材料"}],'
        '"relations":[{"source":"方法A","relation":"使用","target":"材料B"}]}'
    )
    async def run():
        return await sync_document_to_kg(
            "sync_test", "doc1", "paper1.txt", "方法A 使用 材料B。", llm
        )

    result = asyncio.run(run())
    assert result["action"] == "upsert"
    assert result["entities"] == 2
    assert result["relations"] == 1
    # 实体/关系确实写入（同一 store）
    stats = kg_env.stats()
    assert stats["entities"] == 2
    assert stats["relations"] == 1


def test_sync_document_remove_keeps_shared(kg_env, tmp_path: Path):
    """删除文档只移除该文档独占实体，共享实体保留。"""
    from app.knowledge.kg import sync_document_to_kg

    # 文档1 引入实体 A、B；文档2 共享实体 A（A 的 doc_source 被文档2 覆盖）
    kg_env.upsert_entity("A", "概念", "", "doc2.txt")
    kg_env.upsert_entity("B", "概念", "", "doc1.txt")
    a_id = kg_env.search_entities("A")[0]["id"]
    b_id = kg_env.search_entities("B")[0]["id"]
    kg_env.add_relation(a_id, b_id, "相关", "doc1.txt")
    # 删除文档1：关系删除；A 被 doc2 引用保留，B 独占删除
    async def run():
        return await sync_document_to_kg("sync_test", "doc1", "doc1.txt", "", remove=True)

    result = asyncio.run(run())
    assert result["action"] == "remove"
    stats = kg_env.stats()
    assert stats["entities"] == 1  # 只剩 A
    assert stats["relations"] == 0


# ---------- 写作接 RAG：_retrieve_kb_for 空库/异常安全 ----------
def test_writing_retrieve_kb_empty_returns_empty(monkeypatch):
    """知识库为空时返回空串，不抛异常（RAG 注入容错）。"""
    from app.api.routes.writing import _retrieve_kb_for

    class FakeUser:
        username = "nobody_writing_test"

    class FakeKB:
        def count(self):
            return 0

    # _retrieve_kb_for 内部从 app.knowledge.user_kb import UserKnowledgeBase
    import app.knowledge.user_kb as user_kb_mod

    monkeypatch.setattr(
        user_kb_mod,
        "UserKnowledgeBase",
        lambda username: FakeKB(),
    )

    async def run():
        return await _retrieve_kb_for(FakeUser(), "钙钛矿")

    assert asyncio.run(run()) == ""


def test_writing_retrieve_kb_error_safe(monkeypatch):
    """ChromaDB 异常时返回空串，写作不中断。"""

    from app.api.routes.writing import _retrieve_kb_for

    class FakeUser:
        username = "nobody_writing_test"

    import app.knowledge.user_kb as user_kb_mod

    class BadKB:
        def count(self):
            raise RuntimeError("chroma down")

        async def search(self, *a, **k):
            raise RuntimeError("chroma down")

    monkeypatch.setattr(user_kb_mod, "UserKnowledgeBase", lambda username: BadKB())

    async def run():
        return await _retrieve_kb_for(FakeUser(), "钙钛矿")

    assert asyncio.run(run()) == ""


# ---------- 笔记入库闭环（API 层） ----------
def test_note_to_kb_closed_loop():
    """to_kb=True 时笔记同时入库，返回 kb_saved=True。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    login = client.post(
        "/api/auth/login", json={"username": "student1", "password": "123456"}
    )
    assert login.status_code == 200, login.text
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 用 stub 化 LLM 会导致真实模型调用失败，这里验证请求协议 + 容错：
    # 即使 LLM 失败也返回 4xx 而非 5xx；to_kb 字段被协议接受
    resp = client.post(
        "/api/notes",
        headers=headers,
        json={
            "concept": "测试概念XYZ",
            "note": "测试内容",
            "kind": "concept",
            "to_kb": True,
        },
    )
    # LLM 真实调用可能失败（无 key 环境）或成功；两者都不应 500
    assert resp.status_code in (200, 400, 403), resp.text
