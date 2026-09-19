"""评测持久化（save_report）与汇总报告（eval_summary）测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.evaluation.store import EvaluationStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_summary_module():
    from scripts.eval_summary import build_summary, render_text  # noqa: PLC0415

    return build_summary, render_text


def test_save_report_one_shot(tmp_path):
    """save_report 一次写入完整 done 记录，无需 start/finish 两步。"""
    store = EvaluationStore(str(tmp_path / "eval.db"))
    rid = store.save_report(
        "retrieval", "extended",
        {"query_count": 11, "aggregate": {"3": {"recall": 0.5}}},
        config={"k_list": [1, 3, 5]},
    )
    row = store.get(rid)
    assert row["status"] == "done"
    assert row["kind"] == "retrieval"
    assert row["dataset"] == "extended"
    assert row["report"]["query_count"] == 11
    assert row["config"]["k_list"] == [1, 3, 5]
    assert row["completed_at"] is not None


def test_save_report_custom_run_id(tmp_path):
    """可显式指定 run_id（幂等覆盖）。"""
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.save_report("golden", "all", {"passed": 3, "total": 4}, run_id="golden-0813")
    row = store.get("golden-0813")
    assert row["report"]["passed"] == 3


def test_save_report_kind_agent_listable(tmp_path):
    """agent 类型评测可被 list(kind='agent') 查到（admin 接口依赖）。"""
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.save_report("agent", "agent_cases", {"metrics": {"task_success_rate": 0.8}})
    runs = store.list(kind="agent")
    assert len(runs) == 1
    assert runs[0]["kind"] == "agent"


def test_eval_summary_three_kinds(tmp_path):
    """汇总报告覆盖 retrieval/golden/agent 三类，且文本可渲染。"""
    build_summary, render_text = _load_summary_module()
    store = EvaluationStore(str(tmp_path / "eval.db"))
    store.save_report("retrieval", "extended",
                      {"query_count": 11, "aggregate": {"1": {"recall": 0.4, "precision": 0.4,
                                                              "hit_rate": 0.4, "mrr": 0.4, "ndcg": 0.4}}})
    store.save_report("golden", "all", {"passed": 5, "total": 6, "failed": 1})
    store.save_report("agent", "agent_cases", {"metrics": {"task_success_rate": 0.75},
                                               "per_case": [{"ok": True}]})

    summary = build_summary(store.list(limit=10))
    assert "retrieval" in summary and len(summary["retrieval"]) == 1
    assert "golden" in summary and summary["golden"][0]["pass_rate"] == pytest.approx(5 / 6)
    assert "agent" in summary and summary["agent"][0]["task_success_rate"] == 0.75

    text = render_text(summary)
    assert "@1" in text and "Recall" in text
    assert "5/6" in text
    assert "0.75" in text


def test_eval_summary_empty_store(tmp_path):
    """空 store：build_summary 返回空 dict，不抛异常。"""
    build_summary, _ = _load_summary_module()
    store = EvaluationStore(str(tmp_path / "eval.db"))
    assert build_summary(store.list()) == {}
