"""知识图谱 API：从用户知识库构建图谱、查询实体/关系、导出可视化数据。

- POST /api/kg/build            从当前用户知识库批量抽取实体/关系（后台任务）
- GET  /api/kg/build/{task_id}  轮询构建进度
- GET  /api/kg/graph            全图数据（nodes + edges，前端 SVG 可视化）
- GET  /api/kg/entities?q=      实体模糊搜索
- GET  /api/kg/entity/{name}    实体详情 + 相邻关系
- GET  /api/kg/stats            图谱统计（实体数/关系数/类型分布）
- DELETE /api/kg                清空当前用户图谱
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import User, get_current_user
from app.knowledge.kg import build_from_docs, get_kg_store
from app.knowledge.user_kb import UserKnowledgeBase, normalize_kb_name

router = APIRouter(prefix="/kg", tags=["kg"], dependencies=[Depends(get_current_user)])

# 后台构建任务表：task_id -> {"status", "stats", "error"}
_build_tasks: dict[str, dict] = {}
_build_lock = threading.Lock()


async def _run_build(task_id: str, username: str, kb_name: str | None, llm: Any) -> None:
    try:
        store = get_kg_store(username)
        kb = UserKnowledgeBase(username, kb_name)
        docs: list[dict] = []
        seen: set[str] = set()
        for d in kb.list_docs():
            did = d.get("doc_id") or d.get("source") or ""
            if did in seen:
                continue
            seen.add(did)
            docs.append(
                {
                    "doc_id": did,
                    "source": d.get("source") or did,
                    "text": d.get("text") or "",
                }
            )
        stats = await build_from_docs(store, docs, llm)
        with _build_lock:
            _build_tasks[task_id] = {"status": "done", "stats": stats, "error": None}
    except Exception as e:  # noqa: BLE001
        with _build_lock:
            _build_tasks[task_id] = {
                "status": "failed",
                "stats": {},
                "error": str(e)[:300],
            }


def _kb(user: User, kb_name: str | None = None) -> UserKnowledgeBase:
    if kb_name:
        kb_name = normalize_kb_name(kb_name)
    return UserKnowledgeBase(user.username, kb_name)


@router.post("/build")
async def start_build(
    kb_name: str | None = Query(None, description="知识库名（None=默认库）"),
    user: User = Depends(get_current_user),
) -> dict:
    """启动图谱构建（后台任务），立即返回 task_id。"""
    from app.llm.factory import get_aux_llm

    task_id = uuid.uuid4().hex[:12]
    with _build_lock:
        _build_tasks[task_id] = {"status": "running", "stats": {}, "error": None}
    from app.llm.gateway import get_llm_gateway
    from app.llm.router import TaskType

    llm = None  # 通过 gateway 路由（ENTITY_EXTRACTION → 辅助模型）
    asyncio.create_task(_run_build(task_id, user.username, kb_name, llm))
    return {"task_id": task_id, "status": "running", "kb_name": kb_name}


@router.get("/build/{task_id}")
async def get_build(task_id: str) -> dict:
    with _build_lock:
        task = _build_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"构建任务不存在: {task_id}")
    return {"task_id": task_id, **task}


@router.get("/graph")
async def graph(
    limit: int = Query(300, ge=1, le=1000),
    user: User = Depends(get_current_user),
) -> dict:
    return get_kg_store(user.username).graph_data(limit_entities=limit)


@router.get("/entities")
async def search_entities(
    q: str = Query("", description="实体名关键字（空=全部）"),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
) -> dict:
    store = get_kg_store(user.username)
    if q.strip():
        items = store.search_entities(q.strip(), limit)
    else:
        items = store.list_entities(limit)
    return {"username": user.username, "total": len(items), "entities": items}


@router.get("/entity/{name}")
async def entity_detail(name: str, user: User = Depends(get_current_user)) -> dict:
    node = get_kg_store(user.username).query_entity(name)
    if not node:
        raise HTTPException(status_code=404, detail=f"实体不存在: {name}")
    return node


@router.get("/stats")
async def stats(user: User = Depends(get_current_user)) -> dict:
    return get_kg_store(user.username).stats()


@router.delete("")
async def clear(user: User = Depends(get_current_user)) -> dict:
    store = get_kg_store(user.username)
    store.clear()
    return {"username": user.username, "cleared": True}
