"""知识图谱查询工具：让 Agent 检索用户知识图谱中的实体与关系。

- `kg_query`：按关键字搜索实体，返回实体列表（名称/类型/描述/来源文档）。
- `kg_relations`：查询某实体的相邻关系（一跳邻居），返回三元组列表。

配合 RAG 检索使用：图谱擅长回答「A 与 B 是什么关系」这类结构化问题，
而向量检索擅长语义召回，两者互补。
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.knowledge.kg import get_kg_store
from app.tools.base import ToolResult, tool

logger = get_logger("tools.kg")


class KgQueryArgs(BaseModel):
    query: str = Field(..., description="实体名或关键字，如 '钙钛矿'、'Transformer'")


class KgRelationsArgs(BaseModel):
    entity: str = Field(..., description="实体名称，查询它的相邻关系（一跳邻居）")
    max_results: int = Field(20, ge=1, le=50, description="最大返回关系数")


def _store_for_current_user() -> Any:
    """按当前会话用户取图谱存储（ContextVar 未设置时回退 'default'）。"""
    try:
        from app.tools.knowledge_search import current_username

        username = current_username.get() or "default"
    except Exception:  # noqa: BLE001
        username = "default"
    return get_kg_store(username)


@tool(
    name="kg_query",
    description="在用户知识图谱中搜索实体（方法/技术/材料/概念等），返回实体名称、类型与一句话描述。当问题涉及领域概念、方法、材料时优先尝试",
    params=KgQueryArgs,
)
async def kg_query(query: str) -> dict | ToolResult:
    try:
        store = _store_for_current_user()
        items = store.search_entities(query, limit=10)
        if not items:
            return ToolResult(
                success=True,
                output={"query": query, "hits": [], "note": "图谱中未找到匹配实体，可改用 knowledge_search 做全文检索"},
            )
        return {"query": query, "hits": items}
    except Exception as e:  # noqa: BLE001
        logger.warning("kg_query 失败: %s", e)
        return ToolResult(success=False, error=f"知识图谱查询失败: {e}", retryable=False)


@tool(
    name="kg_relations",
    description="查询知识图谱中某实体的相邻关系（如 A 提出 B、C 基于 D），返回三元组列表。适合回答结构化关系问题",
    params=KgRelationsArgs,
)
async def kg_relations(entity: str, max_results: int = 20) -> dict | ToolResult:
    try:
        store = _store_for_current_user()
        node = store.query_entity(entity)
        if not node:
            return ToolResult(
                success=True,
                output={"entity": entity, "hits": [], "note": "图谱中不存在该实体"},
            )
        rels = (node.get("relations") or [])[:max_results]
        return {
            "entity": entity,
            "type": node.get("type"),
            "description": node.get("description"),
            "relations": rels,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("kg_relations 失败: %s", e)
        return ToolResult(success=False, error=f"知识图谱关系查询失败: {e}", retryable=False)


def register_kg_tools(registry: Any) -> None:
    """把图谱工具注册进 ToolRegistry（由 tools/__init__.py 或 deps 调用）。"""
    registry.register(kg_query)
    registry.register(kg_relations)
