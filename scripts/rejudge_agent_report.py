"""Use persisted evaluation conversations to re-score answer quality with an LLM judge."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.agent_eval import load_cases  # noqa: E402
from app.evaluation.answer_eval import evaluate_answer  # noqa: E402


def _conversation(conn: sqlite3.Connection, query: str) -> tuple[str, list[str]]:
    session_id = "eval__" + query[:20]
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,)
    ).fetchall()
    starts = [i for i, (_, role, content) in enumerate(rows) if role == "user" and content == query]
    if not starts:
        return "", []
    turn = rows[starts[-1] :]
    contexts = [str(content or "") for _, role, content in turn if role == "tool" and content]
    answers = [str(content or "") for _, role, content in turn if role == "assistant" and content]
    return (answers[-1] if answers else ""), contexts


async def _main(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.llm.factory import get_llm

    settings = get_settings()
    settings.llm.model = args.model
    judge = get_llm(model=args.model)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cases = load_cases(args.dataset)
    by_query = {case.query: case for case in cases}
    conn = sqlite3.connect(args.memory_db)
    judged = 0
    try:
        for row in report.get("per_case", []):
            case = by_query.get(row.get("query", ""))
            if case is None or row.get("error"):
                continue
            answer, contexts = _conversation(conn, case.query)
            if not answer or not contexts:
                continue
            quality = await evaluate_answer(
                case.query, answer, contexts, case.reference_answer, judge=judge
            )
            row["groundedness"] = quality["groundedness"]
            row["groundedness_method"] = quality["method"]
            row["judge_error"] = quality.get("judge_error", "")
            row["answer_preview"] = answer[:500]
            row["context_count"] = len(contexts)
            judged += 1
            print(f"judged {judged}: {case.query[:35]}")
    finally:
        conn.close()

    scores = [
        float(row["groundedness"])
        for row in report.get("per_case", [])
        if row.get("groundedness") is not None
        and str(row.get("groundedness_method", "")).startswith("llm_judge")
        and not row.get("judge_error")
    ]
    report["metrics"]["groundedness"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    report["metrics"]["groundedness_judged_cases"] = len(scores)
    report["metrics"]["groundedness_judge_model"] = args.model
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}; valid groundedness cases={len(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--memory-db", default="data/memory.db")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--output", required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
