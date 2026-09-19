"""Agent 任务相关模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    goal: str = Field(..., description="研究目标 / 任务描述")
    session_id: str | None = Field(None, description="会话 ID，缺省自动生成")
    use_plan: bool = Field(True, description="是否先规划再执行")


class AgentRunResponse(BaseModel):
    session_id: str
    answer: str
    iterations: int | None = None
