"""无进展后的有界替代检索策略。"""
from app.agent.recovery import RecoveryPolicy


def test_recovery_switches_knowledge_to_arxiv_then_web():
    policy = RecoveryPolicy(budget=2)
    available = {"knowledge_search", "arxiv_search", "web_search"}

    first = policy.next_search(
        failed_tool="knowledge_search",
        query="GraphRAG 图结构",
        available_tools=available,
        blocked_tools={"knowledge_search"},
    )
    assert first and first.tool == "arxiv_search"

    second = policy.next_search(
        failed_tool="arxiv_search",
        query="GraphRAG 图结构",
        available_tools=available,
        blocked_tools={"knowledge_search", "arxiv_search"},
    )
    assert second and second.tool == "web_search"
    assert policy.budget == 0


def test_recovery_stops_when_budget_or_tools_exhausted():
    policy = RecoveryPolicy(budget=1)
    assert policy.next_search(
        failed_tool="knowledge_search",
        query="RAG",
        available_tools={"knowledge_search"},
        blocked_tools={"knowledge_search"},
    ) is None
