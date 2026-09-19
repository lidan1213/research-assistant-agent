"""Agent 任务路由：按目标执行多步任务，支持流式。"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends
from app.core.auth import get_current_user
from sse_starlette.sse import EventSourceResponse

from app.agent.runtime import AgentRuntime
from app.api.deps import get_agent
from app.schemas.agent import AgentRunRequest, AgentRunResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(get_current_user)])


class RouteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    mode: str = "auto"
    use_web: bool = False
    has_image: bool = False


@router.post("/route")
async def inspect_route(req: RouteRequest) -> dict:
    """Return the auditable decision without executing the task."""
    from app.agent.router import get_agent_router

    return get_agent_router().route(
        req.message,
        requested_mode=req.mode,
        images=["present"] if req.has_image else None,
        use_web=req.use_web,
    ).to_dict()


@router.get("/provider-status")
async def provider_status() -> dict:
    """Expose safe main/aux channel classification; never returns credentials."""
    from app.llm.channel import configured_channel

    return {"main": configured_channel(), "aux": configured_channel(use_aux=True)}


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    req: AgentRunRequest, agent: AgentRuntime = Depends(get_agent)
) -> AgentRunResponse:
    session_id = req.session_id or str(uuid.uuid4())
    answer = await agent.run(session_id, req.goal, use_plan=req.use_plan)
    return AgentRunResponse(session_id=session_id, answer=answer)


@router.post("/run/stream")
async def run_agent_stream(
    req: AgentRunRequest, agent: AgentRuntime = Depends(get_agent)
):
    session_id = req.session_id or str(uuid.uuid4())

    async def event_gen():
        async for ev in agent.stream(session_id, req.goal, use_plan=req.use_plan):
            yield {"data": json.dumps(ev.to_dict(), ensure_ascii=False)}

    return EventSourceResponse(event_gen())
