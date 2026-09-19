"""API 鉴权。

- 支持多个 API Key（逗号分隔）。
- 通过中间件在开发环境可关闭（API_KEYS 为空时不校验）。
- 同时提供 `HTTPBearer` 依赖，供需要显式校验的路由使用。
"""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.core.exceptions import AgentError

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str | None:
    """校验请求头中的 Bearer Token；未配置 key 时放行。"""
    settings = get_settings()
    if not settings.api_key_list:
        return None

    if creds is None or creds.credentials not in settings.api_key_list:
        raise AgentError("无效或未提供的 API Key", status_code=401, code="UNAUTHORIZED")
    return creds.credentials
