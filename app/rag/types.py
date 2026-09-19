"""RAG 检索管线类型定义。

RetrievalResult：统一检索结果结构，贯穿 Pipeline 各阶段
（召回 → 融合 → 重排 → 上下文构建），供上层（Agent / 评测 / API）消费。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """单条检索结果。

    - document_id: 文档 ID（评测 qrels 对齐用，去掉扩展名）
    - chunk_id: 分块 ID（doc_id#index）
    - score: 重排后分数（或 RRF 融合分）
    - source: 来源文件名/标识
    - retrieval_method: 命中方式：vector / bm25 / hybrid_rrf / reranked
    - text: 文本内容
    - metadata: 原始元数据透传
    - extra: 附加信息（vector_score / bm25_score 等），不参与序列化
    """

    document_id: str
    chunk_id: str = ""
    score: float = 0.0
    source: str = ""
    retrieval_method: str = "hybrid_rrf"
    text: str = ""
    metadata: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为兼容现有调用方的 dict 形态（retrieve() 返回格式）。"""
        return {
            "doc_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "source": self.source,
            "retrieval_method": self.retrieval_method,
            "text": self.text,
            "metadata": {**self.metadata, "doc_id": self.document_id},
        }

    @classmethod
    def from_candidate(
        cls,
        candidate: dict,
        *,
        method: str = "vector",
        score: float | None = None,
    ) -> "RetrievalResult":
        """从现有候选 dict（vectorstore/BM25 输出）构造。"""
        meta = candidate.get("metadata") or {}
        doc_id = (
            candidate.get("doc_id")
            or meta.get("doc_id")
            or candidate.get("id")
            or ""
        )
        return cls(
            document_id=str(doc_id),
            chunk_id=str(meta.get("chunk_id") or candidate.get("chunk_id") or ""),
            score=score if score is not None else float(candidate.get("score") or 0.0),
            source=str(meta.get("source") or candidate.get("source") or doc_id or "?"),
            retrieval_method=method,
            text=str(candidate.get("text") or ""),
            metadata=meta,
            extra={"raw_score": candidate.get("score")},
        )
