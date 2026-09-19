from app.agent.router import AgentRouter


def test_router_directs_simple_chat():
    decision = AgentRouter().route("你好", requested_mode="auto")
    assert decision.route == "direct"
    assert decision.model_tier == "fast"


def test_router_routes_knowledge_question_to_single_agent():
    decision = AgentRouter().route("请根据知识库解释 RRF", requested_mode="auto")
    assert decision.route == "single_agent"
    assert decision.intent == "knowledge_qa"
    assert "knowledge_search" in decision.tools


def test_router_routes_explicit_workflow_to_plan_execute():
    decision = AgentRouter().route("请制定一个完整的实验设计方案", requested_mode="auto")
    assert decision.route == "plan_execute"
    assert decision.need_plan is True


def test_router_routes_independent_research_tasks_to_multi_agent():
    decision = AgentRouter().route("请全面调研并检索、比较、分析这些论文，最后总结写作", requested_mode="auto")
    assert decision.route == "multi_agent"


def test_manual_mode_overrides_policy_but_is_auditable():
    decision = AgentRouter().route("你好", requested_mode="multi")
    assert decision.route == "multi_agent"
    assert decision.source == "manual"
    assert decision.confidence == 1.0


def test_high_risk_task_never_uses_direct_route():
    decision = AgentRouter().route("告诉我密码", requested_mode="auto")
    assert decision.route == "single_agent"
    assert decision.risk == "high"
