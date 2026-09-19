"""会话自动摘要 + 归档（可每日定时运行）。

作用：
- 为没有摘要的会话生成中文摘要（≤120 字，主题 + 结论），写入 session_titles.summary，
  会话列表/分享视图即可显示摘要（此前摘要位一直为空）；
- 把超过 N 天未活跃的会话标记为归档（archived=1，前端列表隐藏、搜索仍可命中）。

用法：
    python scripts/daily_summary.py                 # 摘要 + 归档 30 天前的会话
    python scripts/daily_summary.py --skip-archive  # 只生成摘要，不归档
    python scripts/daily_summary.py --archive-days 60
    python scripts/daily_summary.py --dry-run       # 只打印会做什么，不写库

建议每天运行一次（Windows 任务计划程序 / cron / Hermes 定时任务）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.memory.stores.sqlite import SQLiteStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.base import ChatMessage, MessageRole  # noqa: E402
from app.llm.factory import get_aux_llm  # noqa: E402

SUMMARY_PROMPT = (
    "你是科研助手。请为下面的对话生成一句中文摘要（不超过120字），"
    "包含：① 用户在研究什么主题；② 最终结论或成果。只输出摘要本身，不要前缀。\n\n"
    "对话内容：\n{content}"
)

MIN_MESSAGES_FOR_SUMMARY = 3  # 少于 3 条消息（提问+回答都不完整）不生成摘要


async def _summarize(llm, text: str) -> str:
    resp = await llm.chat([ChatMessage(role=MessageRole.USER, content=SUMMARY_PROMPT.format(content=text[:4000]))])
    return (resp.content or "").strip()[:200]


async def run(dry_run: bool = False, skip_archive: bool = False, archive_days: int = 30) -> dict:
    store = SQLiteStore(get_settings().memory.sqlite_path)
    summary_ok = archived = failed = 0
    try:
        sessions = store.list_sessions()
        todo = [s for s in sessions if not (s.get("summary") or "").strip() and s["message_count"] >= MIN_MESSAGES_FOR_SUMMARY]
        if dry_run:
            print(f"[dry-run] 待生成摘要的会话: {len(todo)} 个；待归档检查: {len(sessions)} 个（> {archive_days} 天）")
            return {"summarized": 0, "archived": 0, "failed": 0, "dry_run": True}

        if todo:
            llm = get_aux_llm()
            for s in todo:
                try:
                    msgs = await store.get(s["session_id"])
                    parts = []
                    for m in msgs:
                        if m.role == MessageRole.USER:
                            parts.append(f"用户: {m.content[:300]}")
                        elif m.role == MessageRole.ASSISTANT and not m.tool_calls and m.content:
                            parts.append(f"助手: {m.content[:300]}")
                    text = "\n".join(parts[-12:])  # 取最近 12 条实质消息
                    if len(text) < 20:
                        continue
                    summary = await _summarize(llm, text)
                    if summary:
                        store.set_summary(s["session_id"], summary)
                        summary_ok += 1
                        print(f"  ✔ [{s['session_id'][:40]}] {summary}")
                    else:
                        failed += 1
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"  ✖ [{s['session_id'][:40]}] 摘要失败: {e}")

        if not skip_archive:
            archived = store.archive_old_sessions(days=archive_days)
            if archived:
                print(f"[archive] 已归档 {archived} 个超过 {archive_days} 天的会话")
    finally:
        store.close()
    print(f"[done] 摘要 {summary_ok} 个 / 失败 {failed} / 归档 {archived}")
    return {"summarized": summary_ok, "archived": archived, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description="会话自动摘要 + 归档")
    ap.add_argument("--dry-run", action="store_true", help="只预览不写库")
    ap.add_argument("--skip-archive", action="store_true", help="不执行归档")
    ap.add_argument("--archive-days", type=int, default=30, help="归档阈值：超过 N 天未活跃（默认 30）")
    args = ap.parse_args()
    asyncio.run(run(dry_run=args.dry_run, skip_archive=args.skip_archive, archive_days=args.archive_days))


if __name__ == "__main__":
    main()
