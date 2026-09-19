"""知识库相关模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., description="待入库的文档片段")
    metadatas: list[dict[str, Any]] | None = Field(None, description="与 texts 一一对应的元数据")


class QueryRequest(BaseModel):
    query: str = Field(..., description="检索问题")
    top_k: int | None = Field(None, description="召回数量")
    format: bool = Field(True, description="是否返回拼装好的上下文文本")


class BuildDocItem(BaseModel):
    doc_id: str = Field(..., description="稳定文档 id（评测 qrels 据此对齐）")
    text: str = Field(..., description="文档正文")
    source: str = Field("", description="来源标签")


class BuildRequest(BaseModel):
    docs: list[BuildDocItem] = Field(..., description="待入库的语料（带稳定 doc_id）")
