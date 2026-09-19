"""会话级链路追踪（Trace）：记录每次 LLM 调用与工具调用的耗时/成败/成本。

设计：
- SQLite 落盘（data/traces.db），按 (session_id, ts) 可查——admin 排查"哪步卡住/花了多少钱"；
- `current_session_id` ContextVar：WS 层在收到消息时设置，LLM/工具层无需透传参数；
- 记录点：
  - LLM 调用（openai_provider 成功/失败后）→ model / tokens / cost / duration / json_mode
  - 工具调用（agent._run_tool 每次尝试后）→ tool / success / duration / error
- 幂等容错：SQLite 写入失败仅记日志，绝不影响主流程。
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextvars import ContextVar
from typing import Optional

from app.core.logging import get_logger

logger = get_logger("trace")

# 当前会话 ID（WS 层设置）：让 LLM/工具层知道本次调用属于哪个会话
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)
# 单次用户请求链路 ID：贯穿 RAG/LLM/工具 Trace，便于端到端排查。
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)

_LOCK = threading.Lock()


def _db_path() -> str:
    return os.environ.get(
        "TRACE_DB", os.path.join(os.path.dirname(__file__), "..", "..", "data", "traces.db")
    )


class TraceLedger:
    """调用链明细账本（SQLite 落盘）。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or _db_path()
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS traces("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "session_id TEXT, trace_id TEXT, kind TEXT, name TEXT,"
                    "success INTEGER, duration_ms INTEGER,"
                    "detail TEXT, created_at REAL)"
                )
                # 兼容已有 traces.db：SQLite 没有 IF NOT EXISTS for columns。
                cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
                if "trace_id" not in cols:
                    conn.execute("ALTER TABLE traces ADD COLUMN trace_id TEXT")
                conn.commit()
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Trace 表初始化失败（追踪不可用）: {e}")

    def record(
        self,
        kind: str,          # "llm" | "tool"
        name: str,          # 模型名 / 工具名
        *,
        session_id: str | None = None,
        success: bool = True,
        duration_ms: int = 0,
        detail: str = "",
    ) -> None:
        sid = session_id or current_session_id.get() or ""
        tid = current_trace_id.get() or ""
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO traces(session_id, trace_id, kind, name, success, duration_ms, detail, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (sid, tid, kind, name, 1 if success else 0, duration_ms, detail, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Trace 写入失败（忽略）: {e}")

    def query(
        self,
        session_id: str | None = None,
        *,
        limit: int = 200,
        kind: str | None = None,
    ) -> list[dict]:
        """按会话（可空=全部）查询最近调用链。"""
        sql = "SELECT id, session_id, trace_id, kind, name, success, duration_ms, detail, created_at FROM traces"
        conds, params = [], []
        if session_id:
            conds.append("session_id=?")
            params.append(session_id)
        if kind:
            conds.append("kind=?")
            params.append(kind)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            conn = self._conn()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
            return [
                {
                    "id": r[0], "session_id": r[1], "trace_id": r[2], "kind": r[3], "name": r[4],
                    "success": bool(r[5]), "duration_ms": r[6], "detail": r[7],
                    "created_at": r[8],
                }
                for r in rows
            ]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Trace 查询失败（返回空）: {e}")
            return []

    def query_by_trace(self, trace_id: str, limit: int = 500) -> list[dict]:
        """按 trace_id 查询一次请求的完整调用链（正序）。"""
        if not trace_id:
            return []  # 空 trace 无链路意义，直接返回
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, session_id, trace_id, kind, name, success, duration_ms, detail, created_at "
                    "FROM traces WHERE trace_id=? ORDER BY id ASC LIMIT ?",
                    (trace_id, limit),
                ).fetchall()
            finally:
                conn.close()
            return [
                {
                    "id": r[0], "session_id": r[1], "trace_id": r[2], "kind": r[3], "name": r[4],
                    "success": bool(r[5]), "duration_ms": r[6], "detail": r[7],
                    "created_at": r[8],
                }
                for r in rows
            ]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Trace 按 trace_id 查询失败（返回空）: {e}")
            return []

    def session_summary(self, session_id: str) -> dict:
        """某会话的调用统计：次数/耗时/成败/成本。"""
        rows = self.query(session_id, limit=1000)
        llm_calls = [r for r in rows if r["kind"] == "llm"]
        tool_calls = [r for r in rows if r["kind"] == "tool"]
        return {
            "session_id": session_id,
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "failures": sum(1 for r in rows if not r["success"]),
            "total_duration_ms": sum(r["duration_ms"] for r in rows),
            "calls": rows,  # 完整明细（前端展示用）
        }


_LEDGER: Optional[TraceLedger] = None


def get_trace_ledger() -> TraceLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = TraceLedger()
    return _LEDGER


def reset_trace_ledger() -> None:
    """重置（测试用）。"""
    global _LEDGER
    _LEDGER = None
