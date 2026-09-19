"""依赖注入：向路由提供 LLM / 记忆 / 工具 / Agent / 检索器 单例。"""
from __future__ import annotations

from functools import lru_cache

from app.agent.agent import ResearchAgent
from app.memory.conversation import ConversationMemory
from app.agent.plan_execute import PlanExecuteAgent
from app.agent.planner import Planner
from app.agent.runtime import AgentRuntime
from app.graph.multi_agent import MultiAgentSupervisor
from app.graph.single_agent import GraphResearchAgent
from app.knowledge.retriever import Retriever
from app.llm.base import BaseLLM
from app.llm.factory import get_llm
from app.rag.service import RetrievalService
from app.tools.base import ToolRegistry
from app.tools.registry import registry


@lru_cache
def get_memory() -> ConversationMemory:
    return ConversationMemory()


@lru_cache
def get_longterm():
    """跨会话长期记忆（L2）单例：ChromaDB 主存储，SQLite 降级。"""
    from app.memory.longterm import LongTermMemory

    return LongTermMemory()


@lru_cache
def get_registry() -> ToolRegistry:
    return registry


@lru_cache
def get_retriever() -> RetrievalService:
    return Retriever()


@lru_cache
def get_graph_agent() -> GraphResearchAgent:
    return GraphResearchAgent()


@lru_cache
def get_multiagent() -> MultiAgentSupervisor:
    return MultiAgentSupervisor()


def get_agent(
    use_web: bool = True, model: str | None = None,
    allowed_tools: tuple[str, ...] | None = None,
) -> AgentRuntime:
    """按需构建 Agent（内部组件均为缓存单例）。

    use_web=False 时排除 web_search 工具（对应前端「联网检索」开关）。
    model 非空时使用指定模型（用户自选），否则用配置默认。
    """
    tools = registry
    if allowed_tools:
        tools = tools.subset(include=set(allowed_tools))
    if not use_web:
        tools = registry.subset(exclude={"web_search"})
        if allowed_tools:
            tools = tools.subset(include=set(allowed_tools))
    llm: BaseLLM = get_llm(model=model) if model else get_llm()
    return ResearchAgent(
        llm=llm,
        memory=get_memory(),
        tools=tools,
        planner=Planner(llm),
        longterm=get_longterm(),  # 跨会话记忆（L2）
    )


def get_plan_agent(
    use_web: bool = True, allowed_tools: tuple[str, ...] | None = None
) -> PlanExecuteAgent:
    """每个请求创建独立 Plan Agent，避免并发任务共享循环/恢复状态。"""
    tools = registry
    if allowed_tools:
        tools = tools.subset(include=set(allowed_tools))
    if not use_web:
        tools = registry.subset(exclude={"web_search"})
        if allowed_tools:
            tools = tools.subset(include=set(allowed_tools))
    llm: BaseLLM = get_llm()
    return PlanExecuteAgent(llm=llm, planner=Planner(llm), tools=tools)
