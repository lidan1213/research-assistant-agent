from __future__ import annotations

import time

from app.services.admin_metrics import aggregate_cost_trend, aggregate_retrieval_stats


def test_cost_trend_groups_by_day_and_student():
    now = time.time()
    rows = [
        {"created_at": now, "detail": "prompt=100 completion=50", "name": "gpt-4o-mini", "session_id": "alice__s1"},
        {"created_at": now, "detail": "prompt=20 completion=10", "name": "gpt-4o-mini", "session_id": "alice__s2"},
    ]
    result = aggregate_cost_trend(rows, 7)
    assert result["daily"][0]["requests"] == 2
    assert result["daily"][0]["tokens"] == 180
    assert result["by_student"][0]["username"] == "alice"


def test_retrieval_stats_parses_hits_and_failures():
    now = time.time()
    rows = [
        {"created_at": now, "detail": "query='rag' top=0.8 hits=['a.pdf']", "success": True, "duration_ms": 20},
        {"created_at": now, "detail": "query='none' top=0.0 hits=[]", "success": False, "duration_ms": 40},
    ]
    result = aggregate_retrieval_stats(rows, 7)
    assert result["total"] == 2
    assert result["fail_rate"] == 0.5
    assert result["zero_hit_rate"] == 0.5
    assert result["top_docs"] == [{"source": "a.pdf", "hits": 1}]
