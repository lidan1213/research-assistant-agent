"""评测任务持久化存储测试。"""
from __future__ import annotations

from app.evaluation.store import EvaluationStore


def test_store_roundtrip(tmp_path):
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.start("run1", "retrieval", "extended", {"k_list": [1, 3, 5]})
    row = store.get("run1")
    assert row["status"] == "running"
    assert row["dataset"] == "extended"
    assert row["config"] == {"k_list": [1, 3, 5]}

    store.finish("run1", {"aggregate": {"3": {"recall": 0.9}}})
    row = store.get("run1")
    assert row["status"] == "done"
    assert row["report"]["aggregate"]["3"]["recall"] == 0.9
    assert row["completed_at"] is not None


def test_store_fail_and_list(tmp_path):
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.start("r1", "retrieval", "baseline")
    store.fail("r1", "boom")
    store.start("r2", "golden", "golden_set")
    store.finish("r2", {"passed": 5, "total": 5})

    runs = store.list(kind="retrieval")
    assert len(runs) == 1
    assert runs[0]["id"] == "r1"
    assert runs[0]["status"] == "failed"
    assert "boom" in runs[0]["error"]

    all_runs = store.list(limit=10)
    assert len(all_runs) == 2


def test_store_interrupts_stale_running(tmp_path):
    """新实例初始化时，把残留 running 标记为 interrupted（服务重启场景）。"""
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.start("stale1", "retrieval", "extended")
    # 模拟重启：同一路径新建实例
    store2 = EvaluationStore(str(tmp_path / "eval.db"))
    row = store2.get("stale1")
    assert row["status"] == "interrupted"
