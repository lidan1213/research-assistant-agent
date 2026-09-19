"""知识库路由：文档入库与检索（RAG）。仅管理员可访问。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_retriever
from app.core.auth import User, get_current_user, require_admin
from app.rag.service import RetrievalService
from app.schemas.knowledge import BuildRequest, IngestRequest, QueryRequest

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# 知识库管理属于管理员功能：所有路由先过 require_admin
router_deps = [Depends(require_admin)]


@router.post("/ingest", dependencies=router_deps)
async def ingest(req: IngestRequest, retriever: RetrievalService = Depends(get_retriever)):
    count = retriever.ingest(req.texts, req.metadatas)
    return {"ingested": count, "total": retriever.count()}


@router.post("/build", dependencies=router_deps)
async def build(req: BuildRequest, retriever: RetrievalService = Depends(get_retriever)):
    """从带稳定 doc_id 的语料构建本地知识库（默认嵌入为 sentence-transformers，需联网下载权重）。

    若检索器配置了 chunker，则自动分块；否则每篇文档作为一个 chunk。
    """
    docs = [d.model_dump() for d in req.docs]
    count = retriever.ingest_documents(docs, clear=True)
    return {"ingested": count, "total": retriever.count()}


@router.post("/query", dependencies=router_deps)
async def query(req: QueryRequest, retriever: RetrievalService = Depends(get_retriever)):
    candidates = await retriever.retrieve(req.query, req.top_k)
    return {
        "query": req.query,
        "candidates": candidates,
        "context": retriever.format_context(candidates) if req.format else None,
    }


@router.get("/health")
async def health(
    retriever: RetrievalService = Depends(get_retriever),
    _: User = Depends(get_current_user),
):
    """知识库健康探活：登录用户即可访问（前端横幅提示用）。"""
    return {"status": "ok", "documents": retriever.count()}
