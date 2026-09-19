"""RRF（Reciprocal Rank Fusion）：多路召回融合模块。

从 Retriever._rrf 抽取为独立函数，支持任意多路排名列表融合。
"""
from __future__ import annotations

from typing import Iterable


def rrf_fuse(
    ranked_lists: Iterable[list[dict]],
    k: int = 60,
    *,
    key_fn=lambda item: item.get("doc_id") or item.get("id") or "",
    payload_fn=lambda item: item,
) -> list[dict]:
    """融合多个按相关性排序的候选列表（RRF）。

    - ranked_lists: 多个候选列表，每个内部按相关性降序；
    - k: RRF 常数（默认 60，与经典实现一致）；
    - key_fn: 取每个候选的唯一 key（默认 doc_id）；
    - payload_fn: 保留哪个候选作为结果负载（默认原样）。

    返回按 RRF 分数降序的融合列表（保留各候选原始 payload）。
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = key_fn(item)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in payload:
                payload[key] = payload_fn(item)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for key, score in ordered:
        item = payload[key]
        item["rrf_score"] = round(score, 4)
        result.append(item)
    return result


def merge_dedup(
    lists: list[list[dict]],
    *,
    key_fn=lambda item: item.get("doc_id") or item.get("id") or "",
    score_field: str = "score",
) -> list[dict]:
    """多路召回去重合并：同 key 保留最高分（不重排，保持首列表顺序）。"""
    best: dict[str, dict] = {}
    for lst in lists:
        for item in lst:
            key = key_fn(item)
            if not key:
                continue
            if key not in best or (item.get(score_field) or 0) > (best[key].get(score_field) or 0):
                best[key] = item
    return list(best.values())
