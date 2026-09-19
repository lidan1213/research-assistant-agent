"""评测汇总报告：从 EvaluationStore 读取历史评测，输出跨版本指标趋势对比。

用法：
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_summary.py            # 文本汇总
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_summary.py --kind retrieval
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_summary.py --markdown # 表格格式
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_summary.py --json    # 原始 JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KINDS = ("retrieval", "golden", "agent")


def _fmt_timestamp(ts: float | None) -> str:
    if not ts:
        return "-"
    import datetime

    return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def summarize_retrieval(items: list[dict]) -> list[dict]:
    """retrieval：按 dataset 分组，逐次运行输出 @1/@3/@5 关键指标。"""
    rows = []
    for r in sorted(items, key=lambda x: x["created_at"] or 0):
        rep = r.get("report") or {}
        agg = rep.get("aggregate") or {}
        row = {
            "run_id": r["id"], "dataset": r.get("dataset", ""),
            "time": _fmt_timestamp(r.get("created_at")), "queries": rep.get("query_count", 0),
            "k_metrics": {str(k): agg.get(str(k), {}) for k in sorted(agg.keys(), key=int)},
        }
        rows.append(row)
    return rows


def summarize_golden(items: list[dict]) -> list[dict]:
    rows = []
    for r in sorted(items, key=lambda x: x["created_at"] or 0):
        rep = r.get("report") or {}
        total = rep.get("total", 0)
        passed = rep.get("passed", 0)
        rows.append({
            "run_id": r["id"], "dataset": r.get("dataset", "all"),
            "time": _fmt_timestamp(r.get("created_at")),
            "passed": passed, "total": total,
            "pass_rate": (passed / total) if total else None,
            "failed": rep.get("failed", 0),
        })
    return rows


def summarize_agent(items: list[dict]) -> list[dict]:
    rows = []
    for r in sorted(items, key=lambda x: x["created_at"] or 0):
        rep = r.get("report") or {}
        m = rep.get("metrics") or {}
        rows.append({
            "run_id": r["id"], "dataset": r.get("dataset", "agent_cases"),
            "time": _fmt_timestamp(r.get("created_at")),
            "cases": m.get("total_cases") or len(rep.get("per_case") or []),
            "task_success_rate": m.get("task_success_rate"),
            "tool_accuracy": m.get("tool_accuracy") or m.get("tool_selection_accuracy"),
            "avg_cost": m.get("avg_cost"),
        })
    return rows


def build_summary(runs: list[dict]) -> dict:
    by_kind: dict[str, list[dict]] = {}
    for r in runs:
        by_kind.setdefault(r.get("kind", ""), []).append(r)
    out: dict = {}
    if "retrieval" in by_kind:
        out["retrieval"] = summarize_retrieval(by_kind["retrieval"])
    if "golden" in by_kind:
        out["golden"] = summarize_golden(by_kind["golden"])
    if "agent" in by_kind:
        out["agent"] = summarize_agent(by_kind["agent"])
    return out


def render_text(summary: dict) -> str:
    lines: list[str] = []
    for kind in KINDS:
        rows = summary.get(kind)
        if not rows:
            continue
        lines.append(f"\n===== {kind} =====")
        if kind == "retrieval":
            for row in rows:
                lines.append(f"[{row['time']}] {row['dataset']} ({row['queries']}q) run={row['run_id']}")
                for k, m in row["k_metrics"].items():
                    lines.append(
                        f"  @{k}: Recall={m.get('recall', 0):.4f} | Precision={m.get('precision', 0):.4f} | "
                        f"HitRate={m.get('hit_rate', 0):.4f} | MRR={m.get('mrr', 0):.4f} | nDCG={m.get('ndcg', 0):.4f}"
                    )
        elif kind == "golden":
            for row in rows:
                rate = f"{row['pass_rate']:.1%}" if row["pass_rate"] is not None else "-"
                lines.append(f"[{row['time']}] {row['dataset']}: {row['passed']}/{row['total']} ({rate}) run={row['run_id']}")
        elif kind == "agent":
            for row in rows:
                lines.append(
                    f"[{row['time']}] {row['dataset']}: success={row['task_success_rate']:.2f} | "
                    f"tool_acc={row['tool_accuracy']} | cases={row['cases']} run={row['run_id']}"
                )
    return "\n".join(lines)


def render_markdown(summary: dict) -> str:
    md: list[str] = ["# 评测汇总报告", ""]
    for kind in KINDS:
        rows = summary.get(kind)
        if not rows:
            continue
        md.append(f"## {kind}")
        if kind == "retrieval":
            for row in rows:
                md.append(f"### {row['dataset']} · {row['time']} · run `{row['run_id']}`")
                md.append("| K | Recall | Precision | HitRate | MRR | nDCG |")
                md.append("|---|---|---|---|---|---|")
                for k, m in row["k_metrics"].items():
                    md.append(
                        f"| @{k} | {m.get('recall', 0):.4f} | {m.get('precision', 0):.4f} | "
                        f"{m.get('hit_rate', 0):.4f} | {m.get('mrr', 0):.4f} | {m.get('ndcg', 0):.4f} |"
                    )
        elif kind == "golden":
            md.append("| 时间 | 数据集 | 通过 | 通过率 | run_id |")
            md.append("|---|---|---|---|---|")
            for row in rows:
                rate = f"{row['pass_rate']:.1%}" if row["pass_rate"] is not None else "-"
                md.append(f"| {row['time']} | {row['dataset']} | {row['passed']}/{row['total']} | {rate} | `{row['run_id']}` |")
        elif kind == "agent":
            md.append("| 时间 | 数据集 | 成功率 | 工具准确率 | 用例数 | run_id |")
            md.append("|---|---|---|---|---|---|")
            for row in rows:
                md.append(
                    f"| {row['time']} | {row['dataset']} | {row['task_success_rate']:.2f} | "
                    f"{row['tool_accuracy']} | {row['cases']} | `{row['run_id']}` |"
                )
        md.append("")
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测汇总报告")
    parser.add_argument("--kind", choices=KINDS, default=None, help="只汇总某类")
    parser.add_argument("--limit", type=int, default=50, help="每种类型取最近 N 次（默认 50）")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 表格")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args()

    from app.evaluation import get_evaluation_store

    runs = get_evaluation_store().list(kind=args.kind, limit=args.limit)
    if not runs:
        print("EvaluationStore 暂无评测记录。先运行 scripts/eval_retrieval.py / eval_agent.py / eval_golden.py")
        sys.exit(0)

    summary = build_summary(runs)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(render_markdown(summary) if args.markdown else render_text(summary))


if __name__ == "__main__":
    main()
