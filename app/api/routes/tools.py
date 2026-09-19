"""工具管理路由：列出可用工具、直接调用工具（便于调试/编排）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_current_user

from app.api.deps import get_registry
from app.tools.base import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"], dependencies=[Depends(get_current_user)])


@router.get("")
async def list_tools(reg: ToolRegistry = Depends(get_registry)):
    return [t.to_function_schema() for t in reg.list()]


@router.post("/{name}/invoke")
async def invoke_tool(
    name: str, payload: dict, reg: ToolRegistry = Depends(get_registry)
):
    tool = reg.get(name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"工具不存在: {name}")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=422, detail="arguments 必须是对象")
    result = await reg.call(name, json.dumps(arguments, ensure_ascii=False))
    return {"name": name, "success": result.success, "output": result.output, "error": result.error}
