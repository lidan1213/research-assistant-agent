"""Structured Output 统一处理：LLM 输出 → Parser → Validator → Repair。

目标：所有模块（Agent 工具调用解析、写作大纲、KG 抽取、笔记整理）共用一套
容错解析逻辑，不再各自写 try/except。

流程：
    parse()  → 提取 JSON（容忍 markdown 围栏/前后杂讯/截断）
    validate() → 校验结构（必填字段/类型）
    repair() → 已知格式问题自动修复（未闭合括号、引号等）
    parse_and_validate() → 组合入口，返回 (data, error)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


class OutputParseError(ValueError):
    """结构化输出解析失败（携带原因）。"""


def extract_json_text(text: str) -> str:
    """从 LLM 输出中提取 JSON 文本（容忍 ```json 围栏、前后缀杂讯）。"""
    if not text:
        raise OutputParseError("空输出")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    start_a, end_a = t.find("["), t.rfind("]")
    # 优先取更靠前的结构起始（对象或数组），取对应结束
    candidates = []
    if start >= 0 and end > start:
        candidates.append((start, end + 1))
    if start_a >= 0 and end_a > start_a:
        candidates.append((start_a, end_a + 1))
    if not candidates:
        raise OutputParseError("未找到 JSON 结构")
    candidates.sort(key=lambda x: x[0])
    return t[candidates[0][0] : candidates[0][1]].strip()


def _repair_common(text: str) -> str:
    """常见格式修复：尾逗号、未闭合引号、多余反引号。"""
    t = text
    # 去除包裹的 markdown 围栏
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    # 修复尾逗号：{... ,} 或 [...] 中的 "}," 前逗号
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def parse(text: str, *, repair: bool = True) -> Any:
    """解析 LLM 输出为 Python 对象。失败抛 OutputParseError。"""
    if not text or not text.strip():
        raise OutputParseError("空输出")
    raw = text
    for _ in range(2):  # 最多尝试：直接解析 → 提取 JSON → 修复后解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            extracted = extract_json_text(raw)
            return json.loads(extracted)
        except (json.JSONDecodeError, OutputParseError):
            pass
        if repair:
            raw = _repair_common(raw)
        else:
            break
    raise OutputParseError(f"JSON 解析失败: {text[:120]!r}")


def validate(data: Any, validator: Callable[[Any], str | None]) -> tuple[Any, str | None]:
    """校验结构：validator 返回错误信息（None=通过）。"""
    err = validator(data)
    return (data, err)


def parse_and_validate(
    text: str,
    validator: Callable[[Any], str | None] | None = None,
    *,
    repair: bool = True,
) -> tuple[Any, str | None]:
    """组合入口：解析 + 可选校验。返回 (data, error)；解析失败时 error 非空、data=None。"""
    try:
        data = parse(text, repair=repair)
    except OutputParseError as e:
        return None, str(e)
    if validator is not None:
        _, err = validate(data, validator)
        if err:
            return data, err
    return data, None


# ---------- 常用校验器 ----------
def require_fields(*fields: str) -> Callable[[Any], str | None]:
    """校验 dict 包含必填字段。"""

    def _v(data: Any) -> str | None:
        if not isinstance(data, dict):
            return "输出必须是 JSON 对象"
        missing = [f for f in fields if not data.get(f)]
        if missing:
            return f"缺少必填字段: {', '.join(missing)}"
        return None

    return _v


def require_list_of_dicts(data: Any) -> str | None:
    if not isinstance(data, list):
        return "输出必须是 JSON 数组"
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return f"第 {i + 1} 项不是对象"
    return None
