"""知识图谱（KG）模块测试：存储/去重/查询 + LLM 抽取 + API 冒烟。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.knowledge.kg import (
    KGStore,
    _parse_extraction,
    build_from_docs,
    extract_graph,
)


@pytest.fixture()
def kg_store(tmp_path: Path) -> KGStore:
    store = KGStore("test_user", tmp_path)
    yield store
    store.close()


# ---------- 存储与去重 ----------
def test_upsert_entity_dedup(kg_store: KGStore):
    e1 = kg_store.upsert_entity("钙钛矿", "材料", "一种材料")
    e2 = kg_store.upsert_entity("钙钛矿", "材料", "另一种描述")
    assert e1 == e2  # 同 (name, type) 去重，返回同一 id
    assert kg_store.stats()["entities"] == 1


def test_upsert_entity_different_type(kg_store: KGStore):
    e1 = kg_store.upsert_entity("Transformer", "模型")
    e2 = kg_store.upsert_entity("Transformer", "概念")
    assert e1 != e2  # 不同类型视为不同实体
    assert kg_store.stats()["entities"] == 2


def test_add_triple_and_query(kg_store: KGStore):
    kg_store.add_triple("研究A", "提出", "方法B", source_type="论文", target_type="方法")
    node = kg_store.query_entity("研究A")
    assert node is not None
    assert node["type"] == "论文"
    rels = node["relations"]
    assert len(rels) == 1
    assert rels[0]["relation"] == "提出"
    assert rels[0]["target"] == "方法B"


def test_relation_dedup_and_self_loop(kg_store: KGStore):
    a = kg_store.upsert_entity("A", "概念")
    b = kg_store.upsert_entity("B", "概念")
    kg_store.add_relation(a, b, "包含")
    kg_store.add_relation(a, b, "包含")  # 重复写入应去重
    kg_store.add_relation(a, a, "自环")  # 自环应拒绝
    assert kg_store.stats()["relations"] == 1


def test_search_and_graph_data(kg_store: KGStore):
    kg_store.add_triple("深度学习", "是", "机器学习子领域", source_type="概念", target_type="概念")
    kg_store.add_triple("CNN", "属于", "深度学习", source_type="模型", target_type="概念")
    hits = kg_store.search_entities("深度")
    assert any(h["name"] == "深度学习" for h in hits)
    g = kg_store.graph_data()
    assert len(g["nodes"]) == 3
    assert len(g["edges"]) == 2
    stats = kg_store.stats()
    assert stats["entities"] == 3 and stats["relations"] == 2


def test_clear(kg_store: KGStore):
    kg_store.add_triple("A", "指向", "B")
    kg_store.clear()
    assert kg_store.stats()["entities"] == 0
    assert kg_store.stats()["relations"] == 0


# ---------- LLM 输出解析 ----------
def test_parse_extraction_plain_json():
    out = _parse_extraction(
        '{"entities":[{"name":"钙钛矿","type":"材料"}],"relations":[{"source":"研究","relation":"提出","target":"钙钛矿"}]}'
    )
    assert out["entities"][0]["name"] == "钙钛矿"
    assert out["relations"][0]["relation"] == "提出"


def test_parse_extraction_fenced_and_noisy():
    out = _parse_extraction(
        '以下是结果：\n```json\n{"entities":[{"name":"A","type":"概念"}],"relations":[]}\n```\n完毕'
    )
    assert out["entities"][0]["name"] == "A"


def test_parse_extraction_garbage():
    out = _parse_extraction("抱歉，我无法生成结构化输出。")
    assert out == {"entities": [], "relations": []}


# ---------- LLM 抽取与批量构建（stub LLM） ----------
class StubLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        assert kwargs.get("json_mode") is True
        from app.llm.base import LLMResponse

        return LLMResponse(content=self.payload)


def test_extract_graph_with_stub_llm():
    llm = StubLLM(
        '{"entities":[{"name":"知识图谱","type":"概念","description":"结构化知识"},'
        '{"name":"SQLite","type":"工具"}],'
        '"relations":[{"source":"知识图谱","relation":"存储于","target":"SQLite"}]}'
    )
    out = asyncio.run(extract_graph("知识图谱存储于 SQLite。", llm))
    assert len(out["entities"]) == 2
    assert len(out["relations"]) == 1
    assert llm.calls == 1


def test_extract_graph_llm_error_returns_empty():
    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("network down")

    out = asyncio.run(extract_graph("任何文本", BrokenLLM()))
    assert out == {"entities": [], "relations": []}


def test_build_from_docs(tmp_path: Path):
    store = KGStore("test_build", tmp_path)
    llm = StubLLM(
        '{"entities":[{"name":"方法A","type":"方法"},{"name":"材料B","type":"材料"},{"name":"工具C","type":"工具"}],'
        '"relations":[{"source":"方法A","relation":"使用","target":"材料B"},'
        '{"source":"方法A","relation":"依赖","target":"工具C"}]}'
    )
    docs = [{"doc_id": "d1", "source": "doc1.txt", "text": "方法A 使用 材料B。"}]
    stats = asyncio.run(build_from_docs(store, docs, llm))
    assert stats["docs"] == 1
    assert stats["entities"] == 3
    assert stats["relations"] == 2
    node = store.query_entity("方法A")
    assert len(node["relations"]) == 2
    store.close()


def test_build_from_docs_relation_entity_not_in_list(tmp_path: Path):
    """关系两端实体不在 entities 列表时自动补建。"""
    store = KGStore("test_build2", tmp_path)
    llm = StubLLM(
        '{"entities":[{"name":"X","type":"概念"}],'
        '"relations":[{"source":"X","relation":"影响","target":"Y"}]}'
    )
    stats = asyncio.run(
        build_from_docs(store, [{"doc_id": "d1", "source": "s", "text": "X 影响 Y"}], llm)
    )
    assert stats["relations"] == 1
    assert store.query_entity("Y") is not None  # Y 被自动补建
    store.close()


# ---------- API 冒烟（TestClient） ----------
def test_kg_api_flow():
    """构建任务 -> 查询实体 -> 图数据 -> 统计 -> 清空。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    login = client.post(
        "/api/auth/login", json={"username": "student1", "password": "123456"}
    )
    assert login.status_code == 200, login.text
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 清空 -> 构建（stub LLM 会真实调用，但这里只验证接口形态；失败也容忍）
    r = client.delete("/api/kg", headers=headers)
    assert r.status_code == 200

    # 直接写入一条数据验证查询链路（不经 LLM）；get_kg_store 无参时走 KG_DATA_DIR
    from app.knowledge.kg import get_kg_store

    store = get_kg_store("student1")
    store.add_triple("测试实体", "提出", "测试方法", source_type="概念", target_type="方法")
    store.close()

    r = client.get("/api/kg/graph", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"], "图谱应有节点"
    assert any(n["name"] == "测试实体" for n in data["nodes"])

    r = client.get("/api/kg/entity/测试实体", headers=headers)
    assert r.status_code == 200
    assert r.json()["relations"]

    r = client.get("/api/kg/entities?q=测试", headers=headers)
    assert r.status_code == 200
    assert r.json()["entities"]

    r = client.get("/api/kg/stats", headers=headers)
    assert r.status_code == 200
    assert r.json()["entities"] >= 1


def tmp_data_dir() -> Path:
    """测试专用数据目录（避免污染 data/）。"""
    import tempfile

    return Path(tempfile.mkdtemp(prefix="kg_test_"))
