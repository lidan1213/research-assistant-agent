"""统一异常处理。

- 自定义异常层级，便于细化 HTTP 状态码。
- 全局异常处理器，统一返回结构（code / message / detail）。
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger("exception")


class AgentError(Exception):
    """Agent 框架基础异常。"""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ToolError(AgentError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "TOOL_ERROR"


class LLMError(AgentError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "LLM_ERROR"


class MemoryError_(AgentError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "MEMORY_ERROR"


class AuthError(AgentError):
    """认证失败（未登录 / token 无效 / 过期）。"""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "UNAUTHORIZED"


class ForbiddenError(AgentError):
    """权限不足（角色不匹配）。"""

    status_code = status.HTTP_403_FORBIDDEN
    code = "FORBIDDEN"


def _payload(code: str, message: str, detail: object = None) -> dict:
    return {"code": code, "message": message, "detail": detail}


def register_exception_handlers(app: FastAPI) -> None:
    """将异常处理器挂载到应用。"""

    @app.exception_handler(AgentError)
    async def _agent_error_handler(_: Request, exc: AgentError):
        logger.warning(f"{exc.code}: {exc.message} | {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_payload("VALIDATION_ERROR", "请求参数校验失败", exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception):
        logger.exception("未捕获异常")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload("INTERNAL_ERROR", "服务器内部错误", str(exc)),
        )
