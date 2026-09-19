"""Agent 层评测 CLI：工具选择准确率 / 任务成功率 / 成本。

用法：
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_agent.py
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_agent.py --dataset app/evaluation/datasets/agent_cases.json
  PYTHONPATH= .venv/Scripts/python.exe scripts/eval_agent.py --quiet   # CI 用：仅退出码
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.agent_eval import load_cases, run_agent_eval  # noqa: E402


async def _main(args: argparse.Namespace) -> int:
    if args.local_memory:
        from app.config import get_settings

        memory = get_settings().memory
        memory.backend = "sqlite"
        memory.redis_url = ""
    if args.model:
        from app.config import get_settings

        get_settings().llm.model = args.model
    if args.disable_aux:
        # 评测时可固定使用主模型，避免辅助路由模型限流污染能力指标。
        from app.config import get_settings

        get_settings().llm.aux_model = ""
    cases = load_cases(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"=== Agent 评测：{len(cases)} 个用例 ===")

    async def run_one(query: str) -> dict:
        from app.api.deps import get_agent

        agent = get_agent(use_web=False)
        session_id = "eval__" + query[:20]
        # 每个用例从干净会话开始，避免历史评测答案泄漏到当前结果。
        await agent.memory.clear(session_id)
        answer = ""
        tool_calls: list[dict] = []
        contexts: list[str] = []
        iterations = 0
        error_message = ""
        async for event in agent.stream(
            session_id, query, use_plan=False
        ):
            if event.type == "action":
                tool_calls.append({
                    "name": event.data.get("tool", ""),
                    "arguments": event.data.get("arguments") or {},
                    "success": False,
                })
            elif event.type == "observation":
                output = str(event.data.get("output") or "")
                # observation 与 action 按顺序配对；工具执行器的失败会显式返回错误文本。
                pending = next((t for t in tool_calls if not t.get("observed")), None)
                if pending is not None:
                    pending["observed"] = True
                    pending["success"] = not any(
                        marker in output.lower()
                        for marker in ("错误", "异常", "error", "failed", "timeout")
                    )
                    if pending.get("name") in {"knowledge_search", "arxiv_search", "web_search", "pdf_reader"}:
                        contexts.append(output)
            elif event.type == "answer":
                answer = str(event.data.get("content") or "")
            elif event.type == "done":
                iterations = int(event.data.get("iterations") or event.data.get("steps") or 0)
            elif event.type == "error":
                error_message = str(event.data.get("message") or "Agent 执行失败")
        if error_message and not answer:
            raise RuntimeError(error_message)
        for call in tool_calls:
            call.pop("observed", None)
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "iterations": iterations,
            "contexts": contexts,
        }

    report = await run_agent_eval(
        cases,
        lambda: run_one,
        timeout_per_case=args.timeout,
        concurrency=args.concurrency,
    )
    metrics = report["metrics"]
    for case_result, case in zip(report["per_case"], cases):
        mark = "PASS" if case_result["ok"] else "FAIL"
        print(f"  {mark} [{case.tags[0] if case.tags else '?'}] {case.query[:45]}")
    print("\n=== 指标 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入: {args.output}")
    if not args.no_store:
        from app.evaluation import get_evaluation_store

        dataset = Path(args.dataset or "agent_cases").stem
        rid = get_evaluation_store().save_report(
            "agent", dataset, report,
            config={
                "dataset": args.dataset,
                "threshold": args.threshold,
                "timeout_per_case": args.timeout,
                "disable_aux": args.disable_aux,
                "model": args.model or "configured-default",
                "limit": args.limit,
                "local_memory": args.local_memory,
                "concurrency": args.concurrency,
            },
        )
        print(f"已持久化到 EvaluationStore: run_id={rid} (kind=agent, dataset={dataset})")
    return 0 if metrics["task_success_rate"] >= args.threshold else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 层评测")
    parser.add_argument("--dataset", default=None, help="用例 JSON 路径（默认内置样例）")
    parser.add_argument("--output", default="", help="报告输出路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="任务成功率门槛（默认 0.5）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单用例超时秒数")
    parser.add_argument("--disable-aux", action="store_true", help="关闭辅助模型路由，全部使用主模型")
    parser.add_argument("--model", default="", help="本次评测覆盖主模型名称")
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 条（0=全部）")
    parser.add_argument("--no-store", action="store_true", help="不将本次报告写入 EvaluationStore")
    parser.add_argument("--local-memory", action="store_true", help="评测使用 SQLite 会话存储，不依赖 Redis")
    parser.add_argument("--concurrency", type=int, default=1, help="并发评测用例数")
    parser.add_argument("--quiet", action="store_true", help="静默模式（仅退出码）")
    args = parser.parse_args()
    if args.quiet:
        import logging

        logging.disable(logging.CRITICAL)
    code = asyncio.run(_main(args))
    if args.quiet:
        print(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
