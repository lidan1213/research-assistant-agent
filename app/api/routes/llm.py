"""LLM 配置路由：模型列表（用户自选）+ 按模型消费统计（登录用户可见）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.core.auth import User, get_current_user
from app.llm.usage import get_usage_ledger

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/models")
async def list_models(user: User = Depends(get_current_user)) -> dict:
    """可用模型列表 + 当前默认模型（用户自选用）。

    models 来自环境配置 LLM__AVAILABLE_MODELS（逗号分隔）；未配置时
    尝试从供应商 /v1/models 拉取，失败则回退 [当前默认模型]。
    """
    s = get_settings().llm
    configured = [m.strip() for m in (s.available_models or "").split(",") if m.strip()]
    if configured:
        return {
            "models": configured,
            "default": s.model,
            "provider": s.provider,
            "base_url": s.base_url,
        }
    # 未配置：从供应商拉取
    try:
        import httpx

        resp = httpx.get(
            f"{s.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {s.api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
            if models:
                return {
                    "models": models,
                    "default": s.model,
                    "provider": s.provider,
                    "base_url": s.base_url,
                }
    except Exception:  # noqa: BLE001
        pass
    return {
        "models": [s.model] if s.model else [],
        "default": s.model,
        "provider": s.provider,
        "base_url": s.base_url,
    }


@router.get("/usage")
async def llm_usage(user: User = Depends(get_current_user)) -> dict:
    """当前消费统计：总请求/token/成本 + 按模型分账（登录用户可见）。"""
    return get_usage_ledger().summary()
