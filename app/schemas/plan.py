"""先规划后执行（Plan-and-Execute）请求模型。"""

from pydantic import BaseModel, Field


class PlanRunRequest(BaseModel):
    query: str = Field(..., description="用户问题 / 研究目标")
    session_id: str | None = Field(None, description="会话 ID，缺省自动生成")
    use_web: bool = Field(False, description="是否启用联网检索（web_search 工具）")


class PlanResumeRequest(BaseModel):
    resume_token: str = Field(..., description="needs_user_input 事件返回的恢复令牌")
    user_input: str = Field(..., min_length=1, description="用户补充的资料、标题、链接或说明")
    use_web: bool = Field(False, description="恢复执行时是否启用联网检索")
