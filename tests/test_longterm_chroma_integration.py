"""LongTermMemory 的 ChromaDB 主路径集成测试（需本地 ChromaDB 服务端）。

默认被 pyproject addopts 排除（-m 'not chroma'）；显式运行：
    pytest -m chroma tests/test_longterm_chroma_integration.py

覆盖：写入 ChromaDB -> 语义检索 -> category 过滤 -> 会话事实列表。
每个测试使用唯一 session_id 前缀，结束后按 session_id 清理，避免污染共享 collection。
"""
import os

import httpx
import pytest

from app.memory.longterm import LongTermMemory

BASE = os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


def _chroma_available() -> bool:
    try:
        r = httpx.get(f"{BASE}/api/v1/collections", timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.chroma,
    pytest.mark.skipif(not _chroma_available(), reason="需要本地 ChromaDB 服务端（127.0.0.1:8001）"),
]


def _cleanup(session_id: str) -> None:
    """按 session_id 删除该测试写入的 ChromaDB 条目（幂等）。"""
    try:
        lt = LongTermMemory("data/lt_it.db", chroma=True)
        if lt._chroma is not None:  # noqa: SLF001
            base, cid, http = lt._chroma["base"], lt._chroma["cid"], lt._chroma["http"]
            http.post(
                f"{base}/api/v1/collections/{cid}/delete",
                json={"where": {"session_id": {"$eq": session_id}}},
            )
        lt.close()
    except Exception:  # noqa: BLE001
        pass


def test_chroma_semantic_search(tmp_path):
    sid = f"it_{os.getpid()}_semantic"
    _cleanup(sid)
    try:
        lt = LongTermMemory(str(tmp_path / "lt.db"), chroma=True)
        lt.save_session(sid, title="t", summary="s")
        lt.append_fact(sid, "The attention mechanism powers Transformers.", category="finding")
        lt.append_fact(sid, "BERT is a pretrained language model.", category="finding")
        lt.close()

        lt2 = LongTermMemory(str(tmp_path / "lt2.db"), chroma=True)
        # 语义检索：查询词与原文词面不同，验证向量召回而非关键词匹配
        hits = lt2.search_facts("transformer self-attention", k=5)
        assert any("attention mechanism" in h["content"] for h in hits), hits
        # 会话事实列表
        facts = lt2.get_session_facts(sid)
        assert len(facts) == 2, facts
        lt2.close()
    finally:
        _cleanup(sid)


def test_chroma_category_filter(tmp_path):
    sid = f"it_{os.getpid()}_category"
    _cleanup(sid)
    try:
        lt = LongTermMemory(str(tmp_path / "lt.db"), chroma=True)
        lt.append_fact(sid, "Alpha result 1", category="literature")
        lt.append_fact(sid, "Beta observation", category="finding")
        lt.close()

        lt2 = LongTermMemory(str(tmp_path / "lt2.db"), chroma=True)
        by_cat = lt2.search_facts("result", category="literature", k=5)
        assert len(by_cat) == 1 and by_cat[0]["category"] == "literature", by_cat
        lt2.close()
    finally:
        _cleanup(sid)
