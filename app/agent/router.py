"""Unified, auditable task router for the four orchestration strategies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from app.agent.guards import is_simple_task, recommended_tool

Route = Literal["direct", "single_agent", "plan_execute", "multi_agent"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    intent: str
    tools: tuple[str, ...]
    model_tier: str
    need_plan: bool
    risk: str
    confidence: float
    reason: str
    source: str = "policy"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["tools"] = list(self.tools)
        return value


class AgentRouter:
    """Rule-first router; uncertain tasks go to the safe general ReAct path."""

    MANUAL = {
        "react": "single_agent",
        "plan": "plan_execute",
        "multi": "multi_agent",
        "direct": "direct",
    }

    def route(
        self,
        message: str,
        *,
        requested_mode: str = "auto",
        images: list[str] | None = None,
        use_web: bool = False,
    ) -> RouteDecision:
        text = (message or "").strip()
        lower = text.lower()
        intent, tools = self._intent_and_tools(lower, use_web)
        risk = self._risk(lower)

        if requested_mode in self.MANUAL:
            chosen = self.MANUAL[requested_mode]
            return RouteDecision(
                route=chosen, intent=intent, tools=tools,
                model_tier="fast" if chosen == "direct" else "reasoning",
                need_plan=chosen in ("plan_execute", "multi_agent"), risk=risk,
                confidence=1.0, reason=f"用户手动指定 {requested_mode} 模式", source="manual",
            )

        multi_markers = ("系统综述", "全面调研", "多角度", "多方面", "分别调研", "专家协作", "多agent", "多 agent")
        plan_markers = ("实验设计", "研究方案", "实施方案", "分步骤", "制定计划", "完整流程", "综述", "对比分析")
        independent = sum(k in lower for k in ("检索", "分析", "计算", "写作", "总结", "比较"))
        if any(k in lower for k in multi_markers) or independent >= 3:
            return RouteDecision("multi_agent", intent, tools, "reasoning", True, risk, 0.88,
                                 "任务包含多个可独立委派的研究子任务")
        if any(k in lower for k in plan_markers) or len(text) > 300:
            return RouteDecision("plan_execute", intent, tools, "reasoning", True, risk, 0.84,
                                 "任务需要显式步骤和阶段结果汇总")
        if is_simple_task(text, images) and risk == "low":
            return RouteDecision("direct", intent, tools, "fast", False, risk, 0.94,
                                 "短问题且无需工具或多步推理")
        return RouteDecision("single_agent", intent, tools, "reasoning", False, risk, 0.78,
                             "单目标任务，适合 ReAct 按需调用工具")

    @staticmethod
    def _intent_and_tools(text: str, use_web: bool) -> tuple[str, tuple[str, ...]]:
        tool = recommended_tool(text)
        if tool == "arxiv_search":
            return "literature_search", ("arxiv_search", "pdf_reader")
        if tool == "calculator":
            return "calculation", ("calculator",)
        if tool == "code_executor":
            return "code_analysis", ("code_executor", "calculator")
        if tool == "knowledge_search":
            return "knowledge_qa", ("knowledge_search", "kg_query")
        if any(k in text for k in ("写作", "撰写", "润色", "改写")):
            return "academic_writing", ("knowledge_search", "citation")
        if any(k in text for k in ("搜索", "联网", "最新")) and use_web:
            return "web_research", ("web_search",)
        return "general_chat", ()

    @staticmethod
    def _risk(text: str) -> str:
        if any(k in text for k in ("密码", "密钥", "token", "隐私", "删除全部", "清空全部")):
            return "high"
        if any(k in text for k in ("医疗", "诊断", "法律", "投资建议", "财务建议")):
            return "medium"
        return "low"


_router: AgentRouter | None = None


def get_agent_router() -> AgentRouter:
    global _router
    if _router is None:
        _router = AgentRouter()
    return _router
