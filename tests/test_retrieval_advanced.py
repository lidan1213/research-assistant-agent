"""BM25 / 混合检索 / 重排器 单元测试 + 端到端混合检索。"""
from __future__ import annotations

import asyncio
import tempfile

from app.knowledge.bm25 import BM25Index
from app.knowledge.embeddings import HashingEmbedding
from app.knowledge.rerank import CrossEncoderReranker, LexicalReranker, create_reranker
from app.knowledge.retriever import Retriever
from app.knowledge.vectorstore import FAISSVectorStore


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add(["a", "b"], ["机器学习 深度学习 神经网络", "足球 比赛 世界杯"])
    hits = idx.search("机器学习 神经网络", top_k=2)
    assert hits[0]["id"] == "a"


def test_lexical_reranker_prefers_lexical_overlap():
    r = LexicalReranker(w_vector=0.6, w_lexical=0.4)
    cands = [
        {"text": "苹果 公司 手机", "score": 0.9},
        {"text": "苹果 水果 红色", "score": 0.5},
    ]
    out = r.rerank("苹果 水果", cands)
    assert out[0]["text"] == "苹果 水果 红色"


def test_cross_encoder_factory_filters_lexical_only_options():
    """Cross-Encoder 工厂可接收统一配置，不会因 lexical 权重参数报错。"""
    reranker = create_reranker(
        "cross-encoder",
        model_name="test-reranker",
        w_vector=0.7,
        w_lexical=0.3,
    )
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "test-reranker"


def test_cross_encoder_scores_query_document_pairs(monkeypatch):
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            assert pairs == [("科研检索", "无关内容"), ("科研检索", "科研检索方法")]
            return [0.1, 0.9]

    reranker = CrossEncoderReranker("fake")
    monkeypatch.setattr(reranker, "_ensure", lambda: FakeModel())
    out = reranker.rerank(
        "科研检索",
        [{"doc_id": "a", "text": "无关内容"}, {"doc_id": "b", "text": "科研检索方法"}],
    )
    assert out[0]["doc_id"] == "b"
    assert out[0]["rerank_score"] == 0.9


def test_cross_encoder_falls_back_after_model_failure(monkeypatch):
    fallback = LexicalReranker(w_vector=0.0, w_lexical=1.0)
    reranker = CrossEncoderReranker("missing", fallback=fallback)
    monkeypatch.setattr(reranker, "_ensure", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    candidates = [
        {"doc_id": "a", "text": "苹果 手机", "score": 0.9},
        {"doc_id": "b", "text": "苹果 水果", "score": 0.2},
    ]
    assert reranker.rerank("苹果 水果", candidates)[0]["doc_id"] == "b"
    assert reranker._disabled is True


def test_hybrid_retrieve_returns_doc_id():
    tmp = tempfile.mkdtemp()
    store = FAISSVectorStore(index_path=f"{tmp}/idx.faiss")
    r = Retriever(
        embedding=HashingEmbedding(),
        store=store,
        top_k=3,
        rerank_k=3,
        use_bm25=True,
        use_hybrid=True,
    )
    r.ingest(
        ["机器学习 深度学习 神经网络 训练", "足球 比赛 世界杯 进球"],
        [{"doc_id": "m1"}, {"doc_id": "s1"}],
    )

    async def go():
        return await r.retrieve("机器学习 神经网络 训练", top_k=3)

    hits = asyncio.run(go())
    assert any(h.get("doc_id") == "m1" for h in hits)
    # 重排/融合后输出应带重排顺序（列表非空且为 dict）
    assert isinstance(hits[0], dict)


def test_retriever_respects_requested_top_k_above_rerank_k():
    """回归：rerank_k 是候选/重排配置，不得截断调用方请求的最终 top_k。"""
    tmp = tempfile.mkdtemp()
    store = FAISSVectorStore(index_path=f"{tmp}/idx.faiss")
    r = Retriever(
        embedding=HashingEmbedding(),
        store=store,
        top_k=5,
        rerank_k=3,
        use_bm25=True,
        use_hybrid=True,
    )
    r.ingest(
        [f"主题文档 {i} 机器学习 检索 内容" for i in range(6)],
        [{"doc_id": f"d{i}"} for i in range(6)],
    )

    async def go():
        return await r.retrieve("机器学习 检索 内容", top_k=5)

    assert len(asyncio.run(go())) == 5


def test_lexical_reranker_ignores_english_stopwords():
    """英文科研查询不应因 and/is/what 等停用词抬高无关文档。"""
    r = LexicalReranker(w_vector=0.0, w_lexical=1.0)
    out = r.rerank(
        "What is retrieval augmented generation?",
        [
            {"text": "What is this and how does it work?", "score": 0.0},
            {"text": "retrieval augmented generation RAG", "score": 0.0},
        ],
    )
    assert out[0]["text"].startswith("retrieval augmented generation")


def test_retriever_idempotent_bm25():
    tmp = tempfile.mkdtemp()
    store = FAISSVectorStore(index_path=f"{tmp}/idx.faiss")
    r = Retriever(embedding=HashingEmbedding(), store=store, top_k=2, rerank_k=2)
    r.ingest(["人工智能 研究"], [{"doc_id": "x"}])
    r.ingest(["人工智能 研究"], [{"doc_id": "x"}])  # 重复入库
    assert r.count() == 2  # 幂等由调用方保证；此处验证两份独立 chunk 都被索引
