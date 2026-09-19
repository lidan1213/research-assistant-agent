"""任务级循环保护：跨步骤记住无效工具路径，避免下一阶段从零重试。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


def _normalize_result(value: str) -> str:
    """去掉不影响语义的空白和动态数字，使重复结果指纹更稳定。"""
    text = (value or "").strip().lower()
    text = re.sub(r"\d+(?:\.\d+)?(?:ms|s|秒|毫秒)?", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text[:4000]


@dataclass
class TaskLoopGuard:
    """一次完整任务内共享的循环状态。

    第一次结果正常放行；同一工具再次返回相同结果，说明换阶段/换参数仍未
    获得新信息，随即封禁该工具。封禁只在当前任务有效，不影响下一位用户。
    """

    seen_results: set[str] = field(default_factory=set)
    blocked_tools: dict[str, str] = field(default_factory=dict)

    def is_blocked(self, tool_name: str) -> tuple[bool, str]:
        reason = self.blocked_tools.get(tool_name, "")
        return bool(reason), reason

    def record(self, tool_name: str, result: str) -> bool:
        normalized = _normalize_result(result)
        signature = hashlib.sha256(
            f"{tool_name}:{normalized}".encode("utf-8", errors="ignore")
        ).hexdigest()
        if signature in self.seen_results:
            self.blocked_tools[tool_name] = "该工具跨步骤重复返回相同结果，任务没有取得新进展"
            return False
        self.seen_results.add(signature)
        return True

