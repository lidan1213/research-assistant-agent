"""Agent 层评测：工具选择准确率 / 任务成功率 / 迭代与成本指标。

与 RAG 评测（qrels）互补：RAG 评测回答"检索质量"，本模块回答
"Agent 是否选对了工具、是否成功完成任务、花了多少 token/成本"。

用例格式（datasets/agent_cases.json）：
    [
      {
        "query": "计算 123*456",
        "expected_tool": "calculator",        // 期望使用的工具
        "expected_arguments": {"expr": "*"},  // 期望参数的关键键（子串/存在性匹配）
        "expected_result_type": "number",      // number / list / dict / text
        "tags": ["calculator"]
      }
    ]

指标（聚合）：
- task_success_rate: 任务成功率（Agent 完成且无 error 事件）
- tool_call_success_rate: 工具调用成功率（成功工具调用 / 总调用）
- tool_selection_accuracy: 工具选择准确率（首个工具调用 == expected_tool）
- avg_tool_calls_per_task
- avg_iterations_per_task
- avg_tokens_per_request / avg_cost_per_request
- failure_rate
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 期望结果类型 → 判定函数
_RESULT_TYPE_CHECK: dict[str, Callable[[Any], bool]] = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "text": lambda v: isinstance(v, str) and len(v) > 0,
}


@dataclass
class AgentCase:
    query: str
    expected_tool: str = ""
    expected_arguments: dict = field(default_factory=dict)
    expected_result_type: str = "text"
    reference_answer: str = ""
    must_contain: list[str] = field(default_factory=list)
    forbidden_content: list[str] = field(default_factory=list)
    min_answer_chars: int = 1
    correctness_threshold: float = 0.0
    groundedness_threshold: float = 0.0
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentCase":
        return cls(
            query=str(d.get("query", "")),
            expected_tool=str(d.get("expected_tool", "")),
            expected_arguments=dict(d.get("expected_arguments") or {}),
            expected_result_type=str(d.get("expected_result_type", "text")),
            reference_answer=str(d.get("reference_answer", "")),
            must_contain=[str(x) for x in (d.get("must_contain") or [])],
            forbidden_content=[str(x) for x in (d.get("forbidden_content") or [])],
            min_answer_chars=max(0, int(d.get("min_answer_chars", 1))),
            correctness_threshold=float(d.get("correctness_threshold", 0.0)),
            groundedness_threshold=float(d.get("groundedness_threshold", 0.0)),
            tags=[str(t) for t in (d.get("tags") or [])],
        )


@dataclass
class AgentCaseResult:
    """单条用例的执行轨迹（由 runner 填充，指标函数只读）。"""

    query: str = ""
    expected_tool: str = ""
    ok: bool = False
    tool_calls: list[dict] = field(default_factory=list)  # [{name, arguments, success}]
    iterations: int = 0
    error: str = ""
    tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    final_answer: str = ""
    contexts: list[str] = field(default_factory=list)
    answer_correctness: float | None = None
    groundedness: float | None = None
    task_checks: dict[str, bool] = field(default_factory=dict)


# ---------- 指标计算（纯函数） ----------
def compute_agent_metrics(results: list[AgentCaseResult]) -> dict:
    """从用例结果计算聚合指标（纯函数，可单测）。"""
    n = len(results)
    if n == 0:
        return {
            "cases": 0, "task_success_rate": 0.0, "tool_call_success_rate": 0.0,
            "tool_selection_accuracy": 0.0, "avg_tool_calls_per_task": 0.0,
            "avg_iterations_per_task": 0.0, "avg_tokens_per_request": 0.0,
            "avg_cost_per_request": 0.0, "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
            "answer_correctness": 0.0, "groundedness": 0.0, "failure_rate": 0.0,
        }

    task_ok = sum(1 for r in results if r.ok)
    all_tool_calls = sum(len(r.tool_calls) for r in results)
    tool_ok = sum(1 for r in results for t in r.tool_calls if t.get("success"))
    select_ok = sum(
        1 for r in results
        if r.expected_tool and r.tool_calls and (
            r.tool_calls[0].get("name") == r.expected_tool
            or (
                r.expected_tool == "knowledge_search"
                and r.tool_calls[0].get("name") in {"kg_query", "arxiv_search"}
            )
        )
    )
    total_tokens = sum(r.tokens for r in results)
    total_cost = sum(r.cost for r in results)
    total_latency = sum(r.latency_ms for r in results)
    from app.evaluation.answer_eval import latency_percentile

    correctness = [r.answer_correctness for r in results if r.answer_correctness is not None]
    faithfulness = [r.groundedness for r in results if r.groundedness is not None]
    selection_cases = [r for r in results if r.expected_tool]

    return {
        "cases": n,
        "task_success_rate": round(task_ok / n, 4),
        "tool_call_success_rate": round(tool_ok / all_tool_calls, 4) if all_tool_calls else 0.0,
        "tool_selection_accuracy": round(select_ok / len(selection_cases), 4) if selection_cases else 0.0,
        "avg_tool_calls_per_task": round(all_tool_calls / n, 2),
        "avg_iterations_per_task": round(sum(r.iterations for r in results) / n, 2),
        "avg_tokens_per_request": round(total_tokens / n, 1),
        "avg_cost_per_request": round(total_cost / n, 6),
        "avg_latency_ms": round(total_latency / n, 1),
        "p50_latency_ms": latency_percentile([r.latency_ms for r in results], 0.50),
        "p95_latency_ms": latency_percentile([r.latency_ms for r in results], 0.95),
        "answer_correctness": round(sum(correctness) / len(correctness), 4) if correctness else 0.0,
        "groundedness": round(sum(faithfulness) / len(faithfulness), 4) if faithfulness else 0.0,
        "failure_rate": round(1 - task_ok / n, 4),
    }


def check_tool_selection(result: AgentCaseResult, case: AgentCase) -> bool:
    """判定工具选择是否正确（首调用 == expected_tool，参数含 expected 键）。"""
    if not case.expected_tool or not result.tool_calls:
        return True  # 未指定期望工具 → 不判负
    first = result.tool_calls[0]
    acceptable = {case.expected_tool}
    # 知识问答允许 Agent 在文档检索、知识图谱与论文检索之间自主路由。
    if case.expected_tool == "knowledge_search":
        acceptable.update({"kg_query", "arxiv_search"})
    if first.get("name") not in acceptable:
        return False
    args = first.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            args = {}
    for key, val in (case.expected_arguments or {}).items():
        if key not in args:
            return False
        if isinstance(val, str) and val not in str(args.get(key, "")):
            return False
    return True


def check_result_type(answer: str, expected_type: str) -> bool:
    """判定最终答案是否符合期望结果类型。"""
    checker = _RESULT_TYPE_CHECK.get(expected_type)
    if checker is None:
        return True
    # 尝试解析 JSON 化输出；失败按文本判定
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        parsed = answer
    return checker(parsed)


def evaluate_task_success(result: AgentCaseResult, case: AgentCase) -> tuple[bool, dict[str, bool]]:
    """按用例验收条件严格判定任务成功，而非仅检查是否产生文本。"""
    answer = result.final_answer or ""
    acceptable = {case.expected_tool}
    if case.expected_tool == "knowledge_search":
        acceptable.update({"kg_query", "arxiv_search"})
    expected_call = next(
        (t for t in result.tool_calls if t.get("name") in acceptable), None
    ) if case.expected_tool else None
    checks = {
        "no_error": not bool(result.error),
        "answer_present": len(answer.strip()) >= case.min_answer_chars,
        "result_type": check_result_type(answer, case.expected_result_type),
        "tool_selection": check_tool_selection(result, case),
        "tool_execution": bool(expected_call and expected_call.get("success")) if case.expected_tool else True,
        "must_contain": all(token in answer for token in case.must_contain),
        "forbidden_content": all(token not in answer for token in case.forbidden_content),
        "answer_correctness": (
            result.answer_correctness is not None
            and result.answer_correctness >= case.correctness_threshold
        ) if case.correctness_threshold > 0 else True,
        "groundedness": (
            result.groundedness is not None
            and result.groundedness >= case.groundedness_threshold
        ) if case.groundedness_threshold > 0 else True,
    }
    return all(checks.values()), checks


# ---------- 用例数据 ----------
DEFAULT_CASES: list[AgentCase] = [
    AgentCase(query="计算 123*456 等于多少", expected_tool="calculator",
              expected_arguments={"expr": ""}, expected_result_type="number", tags=["calculator"]),
    AgentCase(query="在 arXiv 上搜索 transformer 相关论文", expected_tool="arxiv_search",
              expected_arguments={"query": ""}, expected_result_type="list", tags=["search"]),
    AgentCase(query="用 Python 计算 2 的 10 次方", expected_tool="code_executor",
              expected_result_type="text", tags=["code"]),
]


def load_cases(path: str | Path | None = None) -> list[AgentCase]:
    """加载用例：默认内置；path 指向 JSON 数组文件时加载之。"""
    if not path:
        return DEFAULT_CASES
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    return [AgentCase.from_dict(d) for d in raw if isinstance(d, dict) and d.get("query")]


# ---------- 运行器（依赖注入 Agent，便于测试与 CLI 共用） ----------
async def run_agent_eval(
    cases: list[AgentCase],
    agent_factory: Callable[[], Any],
    *,
    timeout_per_case: float = 120.0,
    judge: Any | None = None,
    concurrency: int = 1,
) -> dict:
    """逐个用例运行 Agent 并采集轨迹。

    agent_factory: 返回一个可调用对象 `await agent(query) -> str`（最终答案）的工厂。
    轨迹（工具调用/迭代数）由调用方通过 result_hook 收集；这里提供默认收集：
    依赖 Agent 的 events 流不可用时退化为仅记录最终答案与耗时。
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_case(case: AgentCase) -> AgentCaseResult:
        r = AgentCaseResult(query=case.query, expected_tool=case.expected_tool)
        t0: float | None = None
        try:
            async with semaphore:
                t0 = time.monotonic()
                agent = agent_factory()
                response = await asyncio.wait_for(agent(case.query), timeout=timeout_per_case)
            # runner 可返回纯文本，也可返回带轨迹的结构化 envelope。
            if isinstance(response, dict):
                r.final_answer = str(response.get("answer") or response.get("final_answer") or "")
                r.tool_calls = list(response.get("tool_calls") or [])
                r.iterations = int(response.get("iterations") or 0)
                r.tokens = int(response.get("tokens") or 0)
                r.cost = float(response.get("cost") or 0.0)
                r.contexts = [str(x) for x in (response.get("contexts") or [])]
            else:
                r.final_answer = str(response or "")

            if case.reference_answer or r.contexts:
                from app.evaluation.answer_eval import evaluate_answer

                quality = await evaluate_answer(
                    case.query, r.final_answer, r.contexts, case.reference_answer, judge=judge
                )
                r.answer_correctness = (
                    float(quality["answer_correctness"]) if case.reference_answer else None
                )
                if case.reference_answer and case.must_contain:
                    # 生成式长答案不应因比参考答案更详细而被 Token F1 过度惩罚。
                    # 人工标注的关键事实覆盖率作为可复现的正确性下限。
                    keypoint_score = sum(
                        1 for item in case.must_contain if item.lower() in r.final_answer.lower()
                    ) / len(case.must_contain)
                    r.answer_correctness = max(r.answer_correctness, keypoint_score)
                r.groundedness = float(quality["groundedness"]) if r.contexts else None
            r.ok, r.task_checks = evaluate_task_success(r, case)
        except asyncio.TimeoutError:
            r.error = f"任务执行超时（>{timeout_per_case}s）"
            r.ok = False
        except Exception as e:  # noqa: BLE001
            r.error = str(e)[:300]
            r.ok = False
        r.latency_ms = int((time.monotonic() - (t0 or time.monotonic())) * 1000)
        return r

    results = list(await asyncio.gather(*(_run_case(case) for case in cases)))

    metrics = compute_agent_metrics(results)
    per_case = [
        {
            "query": r.query,
            "expected_tool": r.expected_tool,
            "ok": r.ok,
            "iterations": r.iterations,
            "tool_calls": r.tool_calls,
            "latency_ms": r.latency_ms,
            "error": r.error,
            "answer_preview": r.final_answer[:120],
            "answer_correctness": r.answer_correctness,
            "groundedness": r.groundedness,
            "task_checks": r.task_checks,
        }
        for r in results
    ]
    return {"metrics": metrics, "per_case": per_case}
