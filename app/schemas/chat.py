"""对话相关模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="用户输入（上限 8000 字）")
    session_id: str | None = Field(None, description="会话 ID，缺省自动生成")
    use_plan: bool = Field(True, description="是否先规划再执行")
    use_web: bool = Field(False, description="是否启用联网检索（web_search 工具）")
    stream: bool = Field(False, description="是否使用 SSE 流式返回（建议用 /chat/stream）")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    model: str | None = None
