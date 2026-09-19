"""RAG 检索离线评测：基于 qrels 计算 Recall/Precision/HitRate/MRR/nDCG。

qrels 格式：
{
  "items": [
    {"query": "...", "relevant_ids": ["doc_id"],
     "relevance": {"doc_id": 3}}
  ]
}

评测直接复用生产 Retriever.retrieve()，因此能真实比较向量检索、BM25 融合与重排链路。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def normalize_doc_id(doc_id: Any) -> str:
    """统一 qrels 与向量库的文档 ID：文件名 ID 去掉末尾扩展名。

    入库路径可能使用 ``arxiv_xxx``，qrels/旧报告可能使用
    ``arxiv_xxx.txt``；二者语义相同，不应因扩展名造成假阴性。
    """
    value = str(doc_id).strip()
    # chunk 级 ID（doc_id#chunk_index）评测时归并到父文档。
    value = value.split("#", 1)[0]
    for suffix in (".txt", ".md", ".pdf"):
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return value


def _ranked_ids(results: list[dict]) -> list[str]:
    ids: list[str] = []
    for item in results:
        metadata = item.get("metadata") or {}
        doc_id = item.get("doc_id") or metadata.get("doc_id") or item.get("id")
        if doc_id is not None and normalize_doc_id(doc_id) not in ids:
            ids.append(normalize_doc_id(doc_id))
    return ids


def metric_at_k(
    ranked_ids: list[str],
    relevant_ids: list[str] | set[str],
    k: int,
    relevance: dict[str, float] | None = None,
) -> dict[str, float]:
    """计算单条 query 的 @K 指标。"""
    ranked = ranked_ids[:k]
    relevant = {normalize_doc_id(x) for x in relevant_ids}
    rel_map = {normalize_doc_id(key): float(value) for key, value in (relevance or {}).items()}
    if not relevant:
        return {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}

    binary_hits = [doc_id in relevant for doc_id in ranked]
    hit_count = sum(binary_hits)
    precision = hit_count / k if k else 0.0
    recall = hit_count / len(relevant)
    hit_rate = 1.0 if hit_count else 0.0
    mrr = 0.0
    for rank, hit in enumerate(binary_hits, 1):
        if hit:
            mrr = 1.0 / rank
            break

    def gain(doc_id: str) -> float:
        return rel_map.get(doc_id, 1.0 if doc_id in relevant else 0.0)

    dcg = sum((2**gain(doc_id) - 1) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked, 1))
    ideal = sorted((rel_map.get(doc_id, 1.0) for doc_id in relevant), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(ideal, 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def load_qrels(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("qrels 必须是数组或包含 items 数组的 JSON 对象")
    normalized = []
    for item in items:
        if not item.get("query"):
            continue
        relevant = item.get("relevant_ids", item.get("relevant_doc_ids", []))
        normalized.append(
            {
                "query": str(item["query"]),
                "relevant_ids": [normalize_doc_id(x) for x in relevant],
                "relevance": item.get("relevance", {}),
                "note": item.get("note", ""),
            }
        )
    return normalized


async def evaluate_retriever(retriever: Any, qrels: list[dict], k_list: list[int] | None = None) -> dict:
    """对 Retriever 执行 qrels 评测并返回可序列化报告。"""
    ks = sorted({int(k) for k in (k_list or [1, 3, 5]) if int(k) > 0})
    per_query: list[dict] = []
    for item in qrels:
        # 一次召回 max(K)，避免每个 K 重复调用 embedding/向量库
        results = await retriever.retrieve(item["query"], top_k=max(ks))
        ranked = _ranked_ids(results)
        metrics = {
            str(k): metric_at_k(ranked, item["relevant_ids"], k, item.get("relevance"))
            for k in ks
        }
        per_query.append(
            {
                "query": item["query"],
                "relevant_ids": item["relevant_ids"],
                "retrieved_ids": ranked[: max(ks)],
                "metrics": metrics,
                "note": item.get("note", ""),
            }
        )

    aggregate = {}
    for k in ks:
        values = [row["metrics"][str(k)] for row in per_query]
        aggregate[str(k)] = {
            name: round(sum(row[name] for row in values) / len(values), 4) if values else 0.0
            for name in ("precision", "recall", "hit_rate", "mrr", "ndcg")
        }
    return {
        "k_list": ks,
        "query_count": len(per_query),
        "aggregate": aggregate,
        "per_query": per_query,
    }


async def evaluate_from_file(retriever: Any, qrels_path: str | Path, k_list: list[int] | None = None) -> dict:
    report = await evaluate_retriever(retriever, load_qrels(qrels_path), k_list)
    report["dataset"] = str(qrels_path)
    return report
