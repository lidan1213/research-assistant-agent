"""Model Router：按任务类型选择主/辅模型（策略表驱动，不写死在业务代码）。

任务类型分类：
- 简单/内部低价值调用（query rewrite、实体抽取、摘要、工具参数提取）→ 辅助模型（便宜）
- 复杂推理/最终答案/用户对话 → 主模型

路由策略通过 TASK_MODEL 表维护，可通过环境变量覆盖（MODEL_ROUTER_OVERRIDES JSON）。
"""
from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """统一任务类型枚举：所有需要调用 LLM 的模块都应标注任务类型。"""

    QUERY_REWRITE = "query_rewrite"          # RAG 查询改写
    ENTITY_EXTRACTION = "entity_extraction"  # 知识图谱实体抽取
    SUMMARIZATION = "summarization"          # 会话摘要/笔记总结
    TOOL_PARAMS = "tool_params"              # 工具参数提取/纠错
    FACT_EXTRACTION = "fact_extraction"      # 长期记忆事实提取
    SELF_EVAL = "self_eval"                  # 答案自评
    TITLE_GEN = "title_gen"                  # 会话标题生成
    SIMPLE_CHAT = "simple_chat"              # 简单闲聊（模型路由判断）
    REASONING = "reasoning"                  # 复杂推理（主链路）
    FINAL_ANSWER = "final_answer"            # 最终答案生成
    DEFAULT = "default"


# 默认路由策略：aux=True 走辅助模型（便宜），False 走主模型
_DEFAULT_TASK_MODEL: dict[str, bool] = {
    TaskType.QUERY_REWRITE.value: True,
    TaskType.ENTITY_EXTRACTION.value: True,
    TaskType.SUMMARIZATION.value: True,
    TaskType.TOOL_PARAMS.value: True,
    TaskType.FACT_EXTRACTION.value: True,
    TaskType.SELF_EVAL.value: True,
    TaskType.TITLE_GEN.value: True,
    TaskType.SIMPLE_CHAT.value: True,
    TaskType.REASONING.value: False,
    TaskType.FINAL_ANSWER.value: False,
    TaskType.DEFAULT.value: False,
}


class ModelRouter:
    """模型路由：任务类型 → 主/辅模型决策。

    策略表可用环境变量 MODEL_ROUTER_OVERRIDES 覆盖（JSON，key=任务类型 value=true/false），
    例如：MODEL_ROUTER_OVERRIDES='{"reasoning": true}' 让复杂推理也走辅助模型。
    """

    def __init__(self, overrides: dict | None = None) -> None:
        self._table = dict(_DEFAULT_TASK_MODEL)
        merged = dict(_DEFAULT_TASK_MODEL)
        merged.update(overrides or self._env_overrides())
        self._table = merged

    @staticmethod
    def _env_overrides() -> dict:
        raw = os.environ.get("MODEL_ROUTER_OVERRIDES", "")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {str(k): bool(v) for k, v in data.items() if isinstance(v, bool)}
        except json.JSONDecodeError:
            return {}

    def uses_aux(self, task_type: str | TaskType | None) -> bool:
        """该任务类型是否应使用辅助模型。"""
        key = task_type.value if isinstance(task_type, TaskType) else (task_type or "default")
        return self._table.get(key, self._table[TaskType.DEFAULT.value])

    def route(
        self, task_type: str | TaskType | None, *, force_main: bool = False
    ) -> dict:
        """返回模型选择决策（供 gateway 使用）。"""
        key = task_type.value if isinstance(task_type, TaskType) else (task_type or "default")
        return {
            "task_type": key,
            "use_aux": not force_main and self.uses_aux(key),
        }

    def set_override(self, task_type: str, use_aux: bool) -> None:
        self._table[task_type] = use_aux


# 全局单例（与 factory 的缓存风格一致）
_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def reset_model_router() -> None:
    global _router
    _router = None
