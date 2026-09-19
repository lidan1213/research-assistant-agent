"""Agent 层评测测试：指标计算 / 工具选择判定 / 结果类型判定 / 用例加载。"""
from __future__ import annotations

from app.evaluation.agent_eval import (
    AgentCase,
    AgentCaseResult,
    check_result_type,
    check_tool_selection,
    compute_agent_metrics,
    evaluate_task_success,
    load_cases,
)


def _result(ok=True, tool_calls=None, iterations=1, tokens=100, cost=0.01, latency=500):
    return AgentCaseResult(
        query="q", expected_tool="calculator", ok=ok,
        tool_calls=tool_calls or [], iterations=iterations,
        tokens=tokens, cost=cost, latency_ms=latency,
    )


def test_metrics_empty():
    m = compute_agent_metrics([])
    assert m["cases"] == 0
    assert m["task_success_rate"] == 0.0


def test_metrics_all_success():
    r = _result(tool_calls=[{"name": "calculator", "success": True}], iterations=2)
    m = compute_agent_metrics([r, r])
    assert m["cases"] == 2
    assert m["task_success_rate"] == 1.0
    assert m["tool_call_success_rate"] == 1.0
    assert m["tool_selection_accuracy"] == 1.0
    assert m["avg_tool_calls_per_task"] == 1.0
    assert m["avg_iterations_per_task"] == 2.0
    assert m["avg_tokens_per_request"] == 100.0
    assert m["avg_cost_per_request"] == 0.01
    assert m["avg_latency_ms"] == 500.0
    assert m["p50_latency_ms"] == 500.0
    assert m["p95_latency_ms"] == 500.0
    assert m["failure_rate"] == 0.0


def test_metrics_mixed_failures():
    ok1 = _result(ok=True, tool_calls=[{"name": "calculator", "success": True}])
    ok2 = _result(ok=True, tool_calls=[{"name": "calculator", "success": True}, {"name": "web_search", "success": False}])
    fail = _result(ok=False, tool_calls=[{"name": "wrong_tool", "success": False}], iterations=5)
    m = compute_agent_metrics([ok1, ok2, fail])
    assert m["task_success_rate"] == round(2 / 3, 4)
    assert m["tool_call_success_rate"] == round(2 / 4, 4)
    assert m["tool_selection_accuracy"] == round(2 / 3, 4)  # fail 的首调用错误
    assert m["failure_rate"] == round(1 / 3, 4)


def test_tool_selection_check():
    case = AgentCase(query="q", expected_tool="calculator", expected_arguments={"expr": ""})
    good = AgentCaseResult(tool_calls=[{"name": "calculator", "arguments": {"expr": "1+1"}}])
    bad_tool = AgentCaseResult(tool_calls=[{"name": "web_search", "arguments": {}}])
    bad_args = AgentCaseResult(tool_calls=[{"name": "calculator", "arguments": {"query": "x"}}])
    assert check_tool_selection(good, case) is True
    assert check_tool_selection(bad_tool, case) is False
    assert check_tool_selection(bad_args, case) is False
    # 未指定期望工具 → 不判负
    assert check_tool_selection(good, AgentCase(query="q")) is True


def test_tool_selection_string_args():
    """参数为 JSON 字符串时也能判定。"""
    case = AgentCase(query="q", expected_tool="calculator", expected_arguments={"expr": ""})
    r = AgentCaseResult(tool_calls=[{"name": "calculator", "arguments": '{"expr": "2*3"}'}])
    assert check_tool_selection(r, case) is True


def test_result_type_check():
    assert check_result_type("42", "number") is True
    assert check_result_type("42.5", "number") is True
    assert check_result_type("abc", "number") is False
    assert check_result_type('["a","b"]', "list") is True
    assert check_result_type('{"a":1}', "dict") is True
    assert check_result_type("有内容", "text") is True
    assert check_result_type("", "text") is False
    assert check_result_type("anything", "unknown_type") is True  # 未知类型不判负


def test_load_cases_default_and_file(tmp_path):
    cases = load_cases()
    assert cases  # 内置样例非空
    assert all(c.query for c in cases)

    p = tmp_path / "cases.json"
    p.write_text(
        '[{"query": "q1", "expected_tool": "calc", "tags": ["t"]},'
        '{"query": "q2"}]',
        encoding="utf-8",
    )
    loaded = load_cases(p)
    assert len(loaded) == 2
    assert loaded[0].expected_tool == "calc"
    assert loaded[1].tags == []


def test_strict_task_success_checks_answer_and_tool_execution():
    case = AgentCase(
        query="查找三篇论文",
        expected_tool="arxiv_search",
        expected_result_type="text",
        must_contain=["论文"],
        forbidden_content=["无法完成"],
        min_answer_chars=4,
    )
    result = AgentCaseResult(
        final_answer="已找到三篇论文",
        tool_calls=[{"name": "arxiv_search", "success": True, "arguments": {}}],
    )
    ok, checks = evaluate_task_success(result, case)
    assert ok is True
    result.tool_calls[0]["success"] = False
    ok, checks = evaluate_task_success(result, case)
    assert ok is False
    assert checks["tool_execution"] is False


def test_metrics_average_answer_quality_and_latency_percentiles():
    rows = [
        AgentCaseResult(ok=True, latency_ms=100, answer_correctness=1.0, groundedness=0.8),
        AgentCaseResult(ok=False, latency_ms=900, answer_correctness=0.5, groundedness=0.6),
    ]
    metrics = compute_agent_metrics(rows)
    assert metrics["answer_correctness"] == 0.75
    assert metrics["groundedness"] == 0.7
    assert metrics["p50_latency_ms"] == 500.0
    assert metrics["p95_latency_ms"] == 860.0
