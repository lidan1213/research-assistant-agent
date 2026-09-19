"""检索质量看板（/api/admin/retrieval-stats）聚合测试。"""
from __future__ import annotations

import os

from app.llm.trace import get_trace_ledger, reset_trace_ledger


def test_retrieval_stats_aggregates(tmp_path, client):
    """聚合：命中次数/失败率/零命中率/文档排行/最近检索。"""
    os.environ["TRACE_DB"] = str(tmp_path / "traces_stats.db")
    reset_trace_ledger()
    ledger = get_trace_ledger()
    # 3 次成功检索（含零命中 1 次）+ 1 次失败
    ledger.record("rag", "retrieve:2", success=True, duration_ms=60,
                  detail="query='量子点 电池' top=0.91 hits=['docA.txt']")
    ledger.record("rag", "retrieve:2", success=True, duration_ms=80,
                  detail="query='钙钛矿' top=0.88 hits=['docB.txt', 'docA.txt']")
    ledger.record("rag", "retrieve:0", success=True, duration_ms=40,
                  detail="query='不存在的内容' top=0.0 hits=[]")
    ledger.record("rag", "retrieve:2", success=False, duration_ms=500,
                  detail="query='超时查询' top=0.0 hits=[]")

    token = client.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()["token"]
    r = client.get("/api/admin/retrieval-stats?days=7", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    j = r.json()
    assert j["total"] == 4
    assert j["failed"] == 1
    assert j["fail_rate"] == 0.25
    assert j["zero_hit_rate"] == 0.5  # 2/4（零命中 + 失败）
    assert j["avg_duration_ms"] == 170.0  # (60+80+40+500)/4
    # 文档命中排行：docA 2 次 > docB 1 次
    assert j["top_docs"][0] == {"source": "docA.txt", "hits": 2}
    assert j["top_docs"][1] == {"source": "docB.txt", "hits": 1}
    # 最近检索包含记录且按时间倒序
    assert len(j["recent"]) == 4
    assert j["recent"][0]["query"] == "超时查询"
    assert j["recent"][0]["success"] is False

    del os.environ["TRACE_DB"]
    reset_trace_ledger()
