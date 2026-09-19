"""答案正确性、忠实度和延迟分位数评测。"""
from __future__ import annotations

import asyncio

import pytest

from app.evaluation.answer_eval import (
    answer_correctness,
    context_metrics,
    evaluate_answer,
    groundedness,
    latency_percentile,
)


def test_answer_correctness_exact_and_partial():
    exact = answer_correctness("Transformer 于 2017 年提出。", "Transformer 于2017年提出")
    assert exact["exact_match"] == 1.0
    assert exact["score"] == 1.0
    partial = answer_correctness("2017 年", "Transformer 于 2017 年提出")
    assert 0 < partial["token_f1"] < 1


def test_groundedness_claim_support():
    report = groundedness(
        "Transformer 于 2017 年提出。月球由奶酪构成。",
        ["论文 Attention Is All You Need 于 2017 年提出 Transformer 架构。"],
        threshold=0.5,
    )
    assert report["total_claims"] == 2
    assert report["supported_claims"] == 1
    assert report["score"] == 0.5


def test_context_metrics():
    metrics = context_metrics("医学影像分割", ["医学影像分割方法", "足球比赛"], "影像分割方法")
    assert metrics["context_precision"] == 0.5
    assert metrics["context_recall"] > 0


def test_latency_percentiles():
    values = [100, 200, 300, 400, 500]
    assert latency_percentile(values, 0.5) == 300.0
    assert latency_percentile(values, 0.95) == 480.0


@pytest.mark.asyncio
async def test_llm_judge_overrides_primary_scores():
    async def judge(_prompt: str) -> str:
        return '{"answer_correctness":0.9,"faithfulness":0.8,"answer_relevance":0.7,"context_relevance":0.6,"reason":"ok"}'

    report = await evaluate_answer("q", "a", ["ctx"], "ref", judge=judge)
    assert report["method"] == "llm_judge+deterministic"
    assert report["answer_correctness"] == 0.9
    assert report["groundedness"] == 0.8


def test_judge_failure_keeps_deterministic_report():
    async def judge(_prompt: str) -> str:
        return "not-json"

    report = asyncio.run(evaluate_answer("问题", "答案", ["答案依据"], "答案", judge=judge))
    assert report["method"] == "deterministic"
    assert "judge_error" in report
