"""RAG 检索管线包：Query → Rewrite → Retrieve → Fuse → Rerank → Context。

模块：
- types: RetrievalResult 统一结果结构
- fusion: RRF 融合 / 多路去重
- pipeline: RetrievalPipeline 编排

与 app/knowledge/ 的分工：
- knowledge/ 是底层存储与算法实现（vectorstore/bm25/rerank/embeddings/chunking）
- rag/ 是检索管线编排与类型抽象（本包）
"""
from app.rag.fusion import merge_dedup, rrf_fuse
from app.rag.pipeline import RetrievalPipeline
from app.rag.service import RetrievalService
from app.rag.types import RetrievalResult

__all__ = ["RetrievalPipeline", "RetrievalResult", "RetrievalService", "merge_dedup", "rrf_fuse"]
