"""RAG Pipeline 测试：RRF 融合 / RetrievalResult / Pipeline 检索与消融开关。"""
from __future__ import annotations

import asyncio

import pytest

from app.rag.fusion import merge_dedup, rrf_fuse
from app.rag.types import RetrievalResult


# ---------- RRF 融合 ----------
def test_rrf_fuse_combines_rankings():
    list_a = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}]
    list_b = [{"doc_id": "d2"}, {"doc_id": "d3"}, {"doc_id": "d4"}]
    fused = rrf_fuse([list_a, list_b], k=60)
    ids = [x["doc_id"] for x in fused]
    assert ids[0] == "d2"  # 两个列表都排前 → RRF 分最高
    assert set(ids) == {"d1", "d2", "d3", "d4"}
    assert all("rrf_score" in x for x in fused)


def test_rrf_fuse_single_list_preserves_order():
    lst = [{"doc_id": "a"}, {"doc_id": "b"}]
    fused = rrf_fuse([lst])
    assert [x["doc_id"] for x in fused] == ["a", "b"]


def test_merge_dedup_keeps_highest_score():
    l1 = [{"doc_id": "a", "score": 0.5}, {"doc_id": "b", "score": 0.4}]
    l2 = [{"doc_id": "a", "score": 0.9}]
    merged = merge_dedup([l1, l2])
    assert len(merged) == 2
    a = next(x for x in merged if x["doc_id"] == "a")
    assert a["score"] == 0.9


# ---------- RetrievalResult ----------
def test_retrieval_result_to_dict():
    r = RetrievalResult(
        document_id="doc1",
        chunk_id="doc1#0",
        score=0.8,
        source="paper.txt",
        retrieval_method="hybrid_rrf",
        text="内容",
        metadata={"chunk_id": "doc1#0"},
    )
    d = r.to_dict()
    assert d["doc_id"] == "doc1"
    assert d["chunk_id"] == "doc1#0"
    assert d["retrieval_method"] == "hybrid_rrf"
    assert d["metadata"]["doc_id"] == "doc1"


def test_retrieval_result_from_candidate():
    cand = {
        "doc_id": "d5",
        "score": 0.7,
        "metadata": {"source": "s.txt", "chunk_id": "d5#2"},
        "text": "x",
    }
    r = RetrievalResult.from_candidate(cand, method="vector")
    assert r.document_id == "d5"
    assert r.source == "s.txt"
    assert r.chunk_id == "d5#2"
    assert r.retrieval_method == "vector"


# ---------- Pipeline（FAISS 内存库，不依赖 ChromaDB） ----------
class FakeEmbedding:
    """简单哈希嵌入：同词向量相近，便于确定性测试。"""

    def embed_query(self, q: str):
        import numpy as np

        vec = np.zeros(8)
        for ch in q:
            vec[hash(ch) % 8] += 1
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed(self, texts: list[str]):
        return [self.embed_query(t) for t in texts]


class FakeStore:
    """极简内存向量库：search 返回全部文档（测试管线编排而非向量质量）。"""

    def __init__(self):
        self._docs: list[dict] = []

    def add(self, vecs, docs):
        self._docs.extend(docs)

    def search(self, vec, k):
        return [{"id": f"doc{i}", "score": 1.0 / (i + 1), "text": d.get("text", ""), "metadata": d.get("metadata", {})} for i, d in enumerate(self._docs[:k])]

    def clear(self):
        self._docs = []


def _pipeline(**kw):
    from app.rag.pipeline import RetrievalPipeline

    return RetrievalPipeline(
        embedding=FakeEmbedding(),
        store=FakeStore(),
        use_rerank=False,  # 测试不依赖重排器
        **kw,
    )


def test_pipeline_retrieve_returns_results():
    p = _pipeline(top_k=3)
    p.ingest_documents(
        [
            {"doc_id": "d1", "text": "钙钛矿太阳能电池效率", "source": "a.txt"},
            {"doc_id": "d2", "text": "机器学习模型训练", "source": "b.txt"},
            {"doc_id": "d3", "text": "钙钛矿制备工艺", "source": "c.txt"},
        ]
    )
    results = asyncio.run(p.retrieve("钙钛矿"))
    assert results
    # 结果结构：doc_id / score / source / retrieval_method / metadata
    for r in results:
        assert r["doc_id"]
        assert "score" in r
        assert r["source"]
        assert r["retrieval_method"] in ("vector", "hybrid_rrf", "reranked")
        assert r["metadata"]["doc_id"] == r["doc_id"]
        assert "retrieval_confidence" in r


def test_pipeline_build_context():
    p = _pipeline()
    results = [
        {"doc_id": "d1", "source": "a.txt", "text": "内容一", "score": 0.9},
        {"doc_id": "d2", "source": "b.txt", "text": "内容二", "score": 0.8},
    ]
    ctx = p.build_context(results)
    assert "[文档1 | 来源:a.txt]" in ctx
    assert "内容一" in ctx and "内容二" in ctx


def test_pipeline_build_context_truncates():
    p = _pipeline()
    results = [{"doc_id": f"d{i}", "source": f"{i}.txt", "text": "字" * 1000, "score": 0.9} for i in range(5)]
    ctx = p.build_context(results, max_chars=500)
    assert len(ctx) <= 700  # 截断生效（含截断标记）
    assert "截断" in ctx


def test_pipeline_disable_bm25():
    """use_bm25=False 时仍可检索（纯向量）。"""
    p = _pipeline(top_k=2, use_bm25=False)
    p.ingest_documents([{"doc_id": "d1", "text": "测试文本内容", "source": "a.txt"}])
    results = asyncio.run(p.retrieve("测试"))
    assert results
    assert all(r["retrieval_method"] == "vector" for r in results)


def test_pipeline_clear_rebuilds_shared_indexes():
    p = _pipeline(top_k=3)
    p.ingest_documents([{"doc_id": "old", "text": "旧知识", "source": "old.txt"}])

    p.ingest_documents(
        [{"doc_id": "new", "text": "新知识", "source": "new.txt"}],
        clear=True,
    )

    assert p.count() == 1
    assert set(p._doc_by_docid) == {"new"}
    results = asyncio.run(p.retrieve("新知识"))
    assert results[0]["doc_id"] == "new"


def test_retriever_compatibility_facade_uses_pipeline_for_ingest():
    from app.knowledge.retriever import Retriever

    r = Retriever(
        embedding=FakeEmbedding(),
        store=FakeStore(),
        top_k=2,
        rerank_k=2,
        use_bm25=True,
        use_hybrid=True,
    )
    r.ingest_documents([{"doc_id": "d1", "text": "统一检索入口", "source": "doc.txt"}])

    assert r.count() == r._pipeline.count() == 1
    assert r.bm25 is r._pipeline.bm25
    assert r._doc_by_docid is r._pipeline._doc_by_docid
    results = asyncio.run(r.retrieve("统一检索入口"))
    assert results[0]["doc_id"] == "d1"
