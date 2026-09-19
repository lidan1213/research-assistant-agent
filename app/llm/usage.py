"""LLM 调用成本 / Token 使用量账本。

目标：
- 每次 chat 调用后自动累计 prompt / completion / total tokens 与估算成本；
- 支持按模型分账（by_model），便于多模型混合部署下的成本归因；
- 线程安全，可在异步并发下安全累加。

成本单价（USD / 1K tokens）为常用模型的内置估算值；命中前缀则采用对应单价，
未命中则回退到 default。生产环境可在配置里覆盖 `COST_TABLE`。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

# (输入单价, 输出单价) 单位：USD / 1K tokens
DEFAULT_COST_TABLE: Dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "deepseek-chat": (0.00027, 0.0011),
    "deepseek-reasoner": (0.00055, 0.00219),
    "default": (0.001, 0.002),
}


@dataclass
class UsageLedger:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    by_model: Dict[str, dict] = field(default_factory=lambda: defaultdict(
        lambda: {"requests": 0, "prompt": 0, "completion": 0, "total": 0, "cost": 0.0}
    ))

    def record(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
        p, c = int(prompt_tokens or 0), int(completion_tokens or 0)
        cost = self._cost(model, p, c)
        with _LOCK:
            self.requests += 1
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += p + c
            self.cost += cost
            b = self.by_model[model]
            b["requests"] += 1
            b["prompt"] += p
            b["completion"] += c
            b["total"] += p + c
            b["cost"] += cost
        return cost

    @staticmethod
    def _cost(model: str, p: int, c: int) -> float:
        rates = DEFAULT_COST_TABLE.get(model)
        if rates is None:
            for k, v in DEFAULT_COST_TABLE.items():
                if k != "default" and model.startswith(k):
                    rates = v
                    break
            rates = rates or DEFAULT_COST_TABLE["default"]
        return p / 1000 * rates[0] + c / 1000 * rates[1]

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost, 6),
            "by_model": {k: dict(v) for k, v in self.by_model.items()},
        }


_LOCK = threading.Lock()
_LEDGER = UsageLedger()


def get_usage_ledger() -> UsageLedger:
    """全局成本/用量账本单例。"""
    return _LEDGER


def reset_usage_ledger() -> None:
    """重置账本（测试或按会话隔离时使用）。"""
    global _LEDGER
    _LEDGER = UsageLedger()
