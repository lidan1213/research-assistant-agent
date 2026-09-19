"""Pure aggregation functions for the administrator dashboard."""
from __future__ import annotations

import re
import time
from ast import literal_eval
from collections import Counter
from datetime import datetime

from app.llm.usage import UsageLedger


def aggregate_cost_trend(rows: list[dict], days: int) -> dict:
    cutoff = time.time() - days * 86400
    daily: dict[str, dict] = {}
    by_student: dict[str, dict] = {}
    for row in rows:
        if row["created_at"] < cutoff:
            continue
        detail = row["detail"] or ""
        prompt_match = re.search(r"prompt=(\d+)", detail)
        completion_match = re.search(r"completion=(\d+)", detail)
        prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0
        completion_tokens = int(completion_match.group(1)) if completion_match else 0
        cost = UsageLedger._cost(row["name"], prompt_tokens, completion_tokens)
        date = datetime.fromtimestamp(row["created_at"]).strftime("%m-%d")
        daily_row = daily.setdefault(date, {"cost": 0.0, "requests": 0, "tokens": 0})
        daily_row["cost"] += cost
        daily_row["requests"] += 1
        daily_row["tokens"] += prompt_tokens + completion_tokens
        username = (row["session_id"] or "").split("__", 1)[0] or "?"
        student = by_student.setdefault(username, {"cost": 0.0, "requests": 0})
        student["cost"] += cost
        student["requests"] += 1
    return {
        "days": days,
        "daily": [{"date": key, **value} for key, value in sorted(daily.items())],
        "by_student": [
            {"username": key, **value}
            for key, value in sorted(by_student.items(), key=lambda item: -item[1]["cost"])
        ],
    }


def aggregate_retrieval_stats(rows: list[dict], days: int) -> dict:
    cutoff = time.time() - days * 86400
    total = failed = zero_hit = 0
    durations = []
    doc_hits: Counter = Counter()
    recent = []
    for row in rows:
        if row["created_at"] < cutoff:
            continue
        total += 1
        failed += not row["success"]
        durations.append(row["duration_ms"] or 0)
        detail = row["detail"] or ""
        top = 0.0
        try:
            match = re.search(r"query=(.*?) top=([\d.]+) hits=(.*)$", detail)
            if match:
                top = float(match.group(2))
                for source in literal_eval(match.group(3)):
                    doc_hits[str(source)] += 1
        except Exception:  # noqa: BLE001
            pass
        zero_hit += top <= 0
        if len(recent) < 20:
            query_match = re.search(r"query=('.*?') top=", detail)
            recent.append(
                {
                    "time": datetime.fromtimestamp(row["created_at"]).strftime("%m-%d %H:%M"),
                    "query": literal_eval(query_match.group(1)) if query_match else detail[:40],
                    "top": top,
                    "success": row["success"],
                    "duration_ms": row["duration_ms"] or 0,
                }
            )
    return {
        "days": days,
        "total": total,
        "failed": failed,
        "fail_rate": round(failed / total, 3) if total else 0.0,
        "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0,
        "zero_hit_rate": round(zero_hit / total, 3) if total else 0.0,
        "top_docs": [{"source": key, "hits": value} for key, value in doc_hits.most_common(10)],
        "recent": recent,
    }
