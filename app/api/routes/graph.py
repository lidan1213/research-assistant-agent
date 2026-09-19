"""LangGraph 与多 Agent 协同路由。"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends
from app.core.auth import get_current_user
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_graph_agent, get_multiagent
from app.graph.multi_agent import MultiAgentSupervisor
from app.graph.single_agent import GraphResearchAgent
from app.schemas.graph import GraphRunRequest, MultiAgentRunRequest

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(get_current_user)])


@router.post("/run")
async def graph_run(
    req: GraphRunRequest, agent: GraphResearchAgent = Depends(get_graph_agent)
) -> dict:
    answer = await agent.run(req.query, req.system_prompt)
    return {"session_id": req.session_id or str(uuid.uuid4()), "answer": answer}


@router.post("/run/stream")
async def graph_stream(
    req: GraphRunRequest, agent: GraphResearchAgent = Depends(get_graph_agent)
):
    async def gen():
        async for ev in agent.stream(req.query, req.system_prompt):
            yield {"data": json.dumps(ev, ensure_ascii=False)}

    return EventSourceResponse(gen())


@router.post("/multiagent/run")
async def multiagent_run(
    req: MultiAgentRunRequest, ma: MultiAgentSupervisor = Depends(get_multiagent)
) -> dict:
    answer = await ma.run(req.query)
    return {"session_id": req.session_id or str(uuid.uuid4()), "answer": answer}


@router.post("/multiagent/run/stream")
async def multiagent_stream(
    req: MultiAgentRunRequest, ma: MultiAgentSupervisor = Depends(get_multiagent)
):
    async def gen():
        async for ev in ma.stream(req.query):
            yield {"data": json.dumps(ev, ensure_ascii=False)}

    return EventSourceResponse(gen())
