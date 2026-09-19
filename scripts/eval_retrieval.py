"""RAG 检索评测 CLI：输出 Recall@K/Precision@K/HitRate@K/MRR@K/nDCG@K。

用法：
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_retrieval.py
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_retrieval.py --qrels data/knowledge/retrieval_qrels.json --k 1 3 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from app.api.deps import get_retriever
    from app.eval_retrieval import evaluate_from_file

    parser = argparse.ArgumentParser(description="RAG qrels 离线评测")
    parser.add_argument("--qrels", default=str(ROOT / "data" / "knowledge" / "retrieval_qrels.json"))
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output", default="", help="可选：保存 JSON 报告路径")
    parser.add_argument("--no-store", action="store_true", help="跳过 EvaluationStore 持久化（默认写入历史）")
    args = parser.parse_args()
    report = asyncio.run(evaluate_from_file(get_retriever(), args.qrels, args.k))
    print(f"=== {report['query_count']} queries | {args.qrels} ===")
    for k in report["k_list"]:
        m = report["aggregate"][str(k)]
        print(
            f"@{k}: Recall={m['recall']:.4f} | Precision={m['precision']:.4f} | "
            f"HitRate={m['hit_rate']:.4f} | MRR={m['mrr']:.4f} | nDCG={m['ndcg']:.4f}"
        )
    for row in report["per_query"]:
        print(f"  [{'OK' if row['retrieved_ids'] else 'MISS'}] {row['query'][:55]} -> {row['retrieved_ids']}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {args.output}")
    if not args.no_store:
        from app.evaluation import get_evaluation_store

        dataset = Path(args.qrels).stem
        rid = get_evaluation_store().save_report(
            "retrieval", dataset, report,
            config={"qrels": str(args.qrels), "k_list": args.k},
        )
        print(f"已持久化到 EvaluationStore: run_id={rid} (kind=retrieval, dataset={dataset})")


if __name__ == "__main__":
    main()
