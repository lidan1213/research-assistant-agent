"""安全计算器：仅允许基础算术，杜绝 eval 注入风险。"""
from __future__ import annotations

import ast
import operator
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolResult, tool

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):  # 数字/字符串常量（仅数字）
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数值常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


class CalcArgs(BaseModel):
    expression: str = Field(..., description="数学表达式，如 (1+2)*3 - 4/2")


@tool(name="calculator", description="安全地计算数学表达式（不含变量/函数调用）", params=CalcArgs)
async def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
        return f"{expression} = {value}"
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"无法计算: {e}")
