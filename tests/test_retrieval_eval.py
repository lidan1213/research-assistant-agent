"""RAG 检索指标与 qrels 评测测试。"""
from __future__ import annotations

import asyncio
import json

from app.eval_retrieval import evaluate_retriever, metric_at_k


def test_metric_at_k_perfect_and_partial():
    m = metric_at_k(["a", "b", "c"], ["a", "c"], 3)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 1.0
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0
    assert 0 < m["ndcg"] <= 1

    miss = metric_at_k(["x", "a"], ["a"], 1)
    assert miss["precision"] == 0.0
    assert miss["recall"] == 0.0
    assert miss["hit_rate"] == 0.0
    assert miss["mrr"] == 0.0


def test_metric_ndcg_relevance_order():
    good = metric_at_k(["high", "low"], ["high", "low"], 2, {"high": 3, "low": 1})
    bad = metric_at_k(["low", "high"], ["high", "low"], 2, {"high": 3, "low": 1})
    assert good["ndcg"] > bad["ndcg"]
    assert good["mrr"] == bad["mrr"] == 1.0


def test_evaluate_retriever_report():
    class StubRetriever:
        async def retrieve(self, query, top_k=None):
            return [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}][:top_k]

    qrels = [
        {"query": "q1", "relevant_ids": ["d1"]},
        {"query": "q2", "relevant_ids": ["d3"]},
    ]
    report = asyncio.run(evaluate_retriever(StubRetriever(), qrels, [1, 3]))
    assert report["query_count"] == 2
    assert report["aggregate"]["1"]["recall"] == 0.5
    assert report["aggregate"]["3"]["recall"] == 1.0
    assert len(report["per_query"]) == 2
    assert report["per_query"][0]["retrieved_ids"] == ["d1", "d2", "d3"]


def test_doc_id_normalization_avoids_extension_false_negative():
    from app.eval_retrieval import metric_at_k, normalize_doc_id

    assert normalize_doc_id("paper.txt") == "paper"
    assert metric_at_k(["paper"], ["paper.txt"], 1)["recall"] == 1.0


def test_qrels_file_is_valid():
    from pathlib import Path

    p = Path("data/knowledge/retrieval_qrels.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["items"]) >= 5
    assert all(item["relevant_ids"] for item in data["items"])
