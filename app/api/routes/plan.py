"""先规划后执行（Plan-and-Execute）模式路由：同步返回 + SSE 流式返回。"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends
from app.core.auth import get_current_user
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_plan_agent
from app.schemas.plan import PlanResumeRequest, PlanRunRequest

router = APIRouter(prefix="/plan", tags=["plan"], dependencies=[Depends(get_current_user)])


@router.post("/run")
async def plan_run(req: PlanRunRequest) -> dict:
    """同步执行：规划 → 逐步执行 → 汇总，返回最终答案。"""
    agent = get_plan_agent(use_web=req.use_web)
    answer = ""
    steps: list[str] = []
    pending: dict | None = None
    async for ev in agent.stream(req.query):
        if ev.type == "plan":
            steps = ev.data.get("steps", [])
        elif ev.type == "answer":
            answer = ev.data.get("content", "")
        elif ev.type == "needs_user_input":
            pending = ev.data
    return {
        "session_id": req.session_id or str(uuid.uuid4()),
        "plan": steps,
        "answer": answer,
        "status": "needs_user_input" if pending else "completed",
        "needs_user_input": pending,
    }


@router.post("/run/stream")
async def plan_stream(req: PlanRunRequest):
    """SSE 流式执行，前端实时渲染 plan / step / action / observation / answer。"""
    agent = get_plan_agent(use_web=req.use_web)

    async def gen():
        async for ev in agent.stream(req.query):
            yield {"data": json.dumps(ev.to_dict(), ensure_ascii=False)}

    return EventSourceResponse(gen())


@router.post("/resume")
async def plan_resume(req: PlanResumeRequest) -> dict:
    """补充信息后从暂停步骤继续，不重新规划已完成部分。"""
    agent = get_plan_agent(use_web=req.use_web)
    answer = ""
    pending: dict | None = None
    async for ev in agent.resume(req.resume_token, req.user_input):
        if ev.type == "answer":
            answer = ev.data.get("content", "")
        elif ev.type == "needs_user_input":
            pending = ev.data
    return {
        "status": "needs_user_input" if pending else "completed",
        "answer": answer,
        "needs_user_input": pending,
    }


@router.post("/resume/stream")
async def plan_resume_stream(req: PlanResumeRequest):
    agent = get_plan_agent(use_web=req.use_web)

    async def gen():
        async for ev in agent.resume(req.resume_token, req.user_input):
            yield {"data": json.dumps(ev.to_dict(), ensure_ascii=False)}

    return EventSourceResponse(gen())
