"""答案级评测：正确性、忠实度、上下文相关性与证据覆盖。

提供两层能力：
1. 确定性离线指标：无需外部模型，适合 CI 回归；
2. 可选 LLM-as-Judge：对开放式回答做语义判断，失败时保留确定性结果。

所有分数统一到 [0, 1]，并显式标注评测方法，避免把启发式分数误称为人工真值。
"""
from __future__ import annotations

import inspect
import json
import math
import re
from collections import Counter
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+", re.I)
_CLAIM_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def _clamp(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def tokenize(text: str) -> list[str]:
    """中英文混合轻量分词：中文字符级，英文/数字词级。"""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def exact_match(answer: str, reference: str) -> float:
    def norm(text: str) -> str:
        return "".join(tokenize(text))

    return 1.0 if norm(answer) and norm(answer) == norm(reference) else 0.0


def token_f1(answer: str, reference: str) -> float:
    """答案与参考答案的 token F1，适合离线确定性回归。"""
    pred, gold = Counter(tokenize(answer)), Counter(tokenize(reference))
    if not pred or not gold:
        return 0.0
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    return _clamp(2 * precision * recall / (precision + recall))


def answer_correctness(answer: str, reference: str) -> dict[str, float]:
    """确定性正确率：Exact Match 与 token F1；综合分偏重语义覆盖 F1。"""
    em = exact_match(answer, reference)
    f1 = token_f1(answer, reference)
    return {"exact_match": em, "token_f1": f1, "score": _clamp(0.2 * em + 0.8 * f1)}


def split_claims(answer: str) -> list[str]:
    return [part.strip() for part in _CLAIM_SPLIT_RE.split(answer or "") if len(tokenize(part)) >= 2]


def _coverage(needle: Iterable[str], haystack: set[str]) -> float:
    tokens = list(needle)
    return sum(1 for t in tokens if t in haystack) / len(tokens) if tokens else 0.0


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> dict[str, Any]:
    """可解释的启发式忠实度：回答事实句中被检索上下文支持的比例。"""
    claims = split_claims(answer)
    context_tokens = set(tokenize("\n".join(contexts or [])))
    details = []
    for claim in claims:
        support = _coverage(tokenize(claim), context_tokens)
        details.append({"claim": claim, "support": round(support, 4), "grounded": support >= threshold})
    supported = sum(1 for row in details if row["grounded"])
    return {
        "score": _clamp(supported / len(details)) if details else 0.0,
        "supported_claims": supported,
        "total_claims": len(details),
        "threshold": threshold,
        "claims": details,
    }


def context_metrics(question: str, contexts: list[str], reference: str = "") -> dict[str, float]:
    """上下文相关率与参考答案覆盖率（确定性诊断指标）。"""
    q_tokens = set(tokenize(question))
    relevant = sum(1 for ctx in contexts if q_tokens and q_tokens.intersection(tokenize(ctx)))
    context_precision = relevant / len(contexts) if contexts else 0.0
    reference_tokens = tokenize(reference)
    context_tokens = set(tokenize("\n".join(contexts or [])))
    context_recall = _coverage(reference_tokens, context_tokens) if reference_tokens else 0.0
    return {
        "context_precision": _clamp(context_precision),
        "context_recall": _clamp(context_recall),
    }


def latency_percentile(values: list[int | float], percentile: float) -> float:
    """线性插值分位数，与常见统计包默认算法一致。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lo, hi = math.floor(pos), math.ceil(pos)
    value = ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return round(value, 1)


def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip().replace("```json", "```")
    if "```" in cleaned:
        cleaned = cleaned.split("```", 2)[1]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge 未返回 JSON 对象")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Judge 结果必须是 JSON 对象")
    return data


async def _call_judge(judge: Any, prompt: str) -> str:
    """兼容 async callable、BaseLLM.chat 与 LangChain ainvoke。"""
    if callable(judge):
        value = judge(prompt)
    elif hasattr(judge, "chat"):
        from app.llm.base import ChatMessage, MessageRole

        value = judge.chat([ChatMessage(role=MessageRole.USER, content=prompt)], json_mode=True)
    elif hasattr(judge, "ainvoke"):
        value = judge.ainvoke(prompt)
    else:
        raise TypeError("judge 必须是 callable、BaseLLM 或 LangChain Runnable")
    if inspect.isawaitable(value):
        value = await value
    return str(getattr(value, "content", value) or "")


async def evaluate_answer(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str = "",
    *,
    judge: Any | None = None,
) -> dict[str, Any]:
    """生成答案级完整报告；Judge 失败时不影响确定性指标。"""
    correctness = answer_correctness(answer, reference_answer) if reference_answer else {
        "exact_match": 0.0, "token_f1": 0.0, "score": 0.0
    }
    faith = groundedness(answer, contexts)
    report: dict[str, Any] = {
        "method": "deterministic",
        "answer_correctness": correctness["score"],
        "exact_match": correctness["exact_match"],
        "token_f1": correctness["token_f1"],
        "groundedness": faith["score"],
        "groundedness_detail": faith,
        **context_metrics(question, contexts, reference_answer),
    }
    if judge is None:
        return report

    prompt = f"""你是严格的 RAG 答案评测器。只依据给定问题、参考答案和检索上下文评分。
输出 JSON，不要解释：
{{"answer_correctness":0到1,"faithfulness":0到1,"answer_relevance":0到1,"context_relevance":0到1,"reason":"简短原因"}}

问题：{question}
参考答案：{reference_answer or '未提供；此时 correctness 仅评价是否合理回答问题'}
检索上下文：
{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(contexts))}

待评回答：{answer}
"""
    try:
        judged = _extract_json(await _call_judge(judge, prompt))
        judge_scores = {
            "answer_correctness": _clamp(judged.get("answer_correctness")),
            "faithfulness": _clamp(judged.get("faithfulness")),
            "answer_relevance": _clamp(judged.get("answer_relevance")),
            "context_relevance": _clamp(judged.get("context_relevance")),
            "reason": str(judged.get("reason", ""))[:500],
        }
        report.update({
            "method": "llm_judge+deterministic",
            "answer_correctness": judge_scores["answer_correctness"],
            "groundedness": judge_scores["faithfulness"],
            "judge": judge_scores,
        })
    except Exception as exc:  # noqa: BLE001
        report["judge_error"] = str(exc)[:300]
    return report
