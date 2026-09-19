"""LLM 成本 / token 用量账本测试。"""
from __future__ import annotations

from app.llm.usage import UsageLedger, get_usage_ledger, reset_usage_ledger


def test_usage_ledger_records_and_summarizes():
    reset_usage_ledger()
    ledger = get_usage_ledger()

    reset_usage_ledger()
    cost = ledger.record("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    s = ledger.summary()

    assert s["requests"] == 1
    assert s["prompt_tokens"] == 1000
    assert s["completion_tokens"] == 500
    assert s["total_tokens"] == 1500
    assert s["cost_usd"] > 0
    assert "gpt-4o" in s["by_model"]
    assert cost > 0


def test_usage_ledger_model_prefix_match():
    ledger = UsageLedger()
    # deepseek-chat 命中前缀单价
    ledger.record("deepseek-chat-1234", prompt_tokens=1000, completion_tokens=0)
    assert "deepseek-chat-1234" in ledger.by_model


def test_usage_ledger_unknown_model_falls_back():
    ledger = UsageLedger()
    ledger.record("some-unknown-model", prompt_tokens=100, completion_tokens=100)
    # 回退到 default 单价，成本为正
    assert ledger.cost > 0
