"""LangGraph / 多 Agent 模块测试（仅验证图构建，不触网）。"""
from app.graph.langchain_tools import build_langchain_tools
from app.graph.multi_agent import MultiAgentSupervisor
from app.graph.single_agent import GraphResearchAgent

EXPECTED_TOOLS = {
    "calculator",
    "web_search",
    "arxiv_search",
    "pdf_reader",
    "code_executor",
    "citation",
}


def test_build_langchain_tools():
    tools = build_langchain_tools()
    assert {t.name for t in tools} >= EXPECTED_TOOLS


def test_graph_agent_builds():
    agent = GraphResearchAgent()
    assert agent._graph is not None


def test_multiagent_builds():
    ma = MultiAgentSupervisor()
    assert ma._graph is not None
    assert set(ma.members) == {"Researcher", "Analyst", "Writer"}
