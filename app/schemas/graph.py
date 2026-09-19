"""LangGraph / 多 Agent 相关模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GraphRunRequest(BaseModel):
    query: str = Field(..., description="用户问题 / 研究目标")
    session_id: str | None = Field(None, description="会话 ID，缺省自动生成")
    system_prompt: str | None = Field(None, description="覆盖默认系统提示词")


class MultiAgentRunRequest(BaseModel):
    query: str = Field(..., description="用户问题 / 研究目标")
    session_id: str | None = Field(None, description="会话 ID，缺省自动生成")
