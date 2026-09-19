"""Agent 回归评测集（Golden Set）：固定问题验证回答质量，防止行为退化。

用法:
  # 全量运行（默认）
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_golden.py

  # 只看某类问题（tool=calculator / kb / search / plain）
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_golden.py --category calculator

  # 静默模式（仅退出码，CI 用）：0=全过 1=有失败
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_golden.py --quiet

用例与运行逻辑在 app/eval_golden.py（与管理端一键评测共用）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval_golden import GOLDEN_CASES, run_all  # noqa: E402


async def _main(args: argparse.Namespace) -> int:
    report = await run_all(str(ROOT / "data" / "golden_eval.db"), args.category)
    for r in report["results"]:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} [{r['category']}] {r['question'][:40]}…")
        if not r["ok"]:
            print(f"      {r['detail']}")
    print(f"\n=== Golden Set 报告：{report['passed']}/{report['total']} 通过 ===")
    if report["failed"]:
        print(f"失败 {report['failed']} 项（可接受：模型回答风格差异/网络波动）")
    if not args.no_store:
        from app.evaluation import get_evaluation_store

        dataset = args.category or "all"
        rid = get_evaluation_store().save_report(
            "golden", dataset, report,
            config={"category": args.category},
        )
        print(f"已持久化到 EvaluationStore: run_id={rid} (kind=golden, dataset={dataset})")
    return 0 if report["failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Set 回归评测")
    parser.add_argument("--category", default=None, help="只跑某类：calculator/kb/search/plain")
    parser.add_argument("--quiet", action="store_true", help="静默模式（仅退出码）")
    parser.add_argument("--no-store", action="store_true", help="跳过 EvaluationStore 持久化（默认写入历史）")
    args = parser.parse_args()
    if args.quiet:
        asyncio.run(_main(args))
    else:
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
