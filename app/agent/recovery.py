"""Agent 恢复策略：无效路径被 LoopGuard 阻止后，选择尚未尝试的替代路线。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


SEARCH_LADDER = ("knowledge_search", "arxiv_search", "web_search")


@dataclass(frozen=True)
class RecoveryDecision:
    tool: str
    arguments: str
    reason: str


@dataclass
class RecoveryPolicy:
    """任务级、有预算的恢复控制器，避免恢复流程自身形成新循环。"""

    budget: int = 2
    attempted: set[str] = field(default_factory=set)

    def next_search(
        self,
        *,
        failed_tool: str,
        query: str,
        available_tools: set[str],
        blocked_tools: set[str],
    ) -> RecoveryDecision | None:
        if self.budget <= 0:
            return None

        # 从失败工具的下一层开始；未知检索工具则从知识库开始。
        try:
            start = SEARCH_LADDER.index(failed_tool) + 1
        except ValueError:
            start = 0
        candidates = SEARCH_LADDER[start:] + SEARCH_LADDER[:start]

        for name in candidates:
            if name == failed_tool or name not in available_tools or name in blocked_tools:
                continue
            key = f"{name}:{query.strip().lower()}"
            if key in self.attempted:
                continue
            self.attempted.add(key)
            self.budget -= 1
            if name == "knowledge_search":
                args = {"query": query, "top_k": 3}
            elif name == "arxiv_search":
                args = {"query": query, "max_results": 5, "download": False}
            else:
                args = {"query": query, "num_results": 5}
            return RecoveryDecision(
                tool=name,
                arguments=json.dumps(args, ensure_ascii=False),
                reason=f"{failed_tool} 未取得新信息，切换到 {name}",
            )
        return None

