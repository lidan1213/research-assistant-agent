"""评测任务持久化：SQLite 落盘，服务重启后历史评测仍可查询与对比。

表 evaluation_runs：
  id            TEXT PK（uuid）
  kind          TEXT   -- retrieval | golden | agent | answer
  dataset       TEXT   -- baseline | extended | chinese | agent 数据集名等
  config_json   TEXT   -- embedding/reranker/top_k 等配置快照
  status        TEXT   -- running | done | failed | interrupted
  report_json   TEXT
  error         TEXT
  created_at    REAL
  completed_at  REAL
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

_LOCK = threading.Lock()


def _default_db_path() -> str:
    return os.environ.get(
        "EVAL_DB",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation_runs.db"),
    )


class EvaluationStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or _default_db_path()
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with _LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS evaluation_runs("
                    "id TEXT PRIMARY KEY, kind TEXT, dataset TEXT, config_json TEXT,"
                    "status TEXT, report_json TEXT, error TEXT,"
                    "created_at REAL, completed_at REAL)"
                )
                conn.commit()
                # 重启后把残留 running 标记为 interrupted
                conn.execute(
                    "UPDATE evaluation_runs SET status='interrupted' WHERE status='running'"
                )
                conn.commit()
            finally:
                conn.close()

    def start(self, run_id: str, kind: str, dataset: str = "", config: dict | None = None) -> None:
        with _LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO evaluation_runs"
                    "(id, kind, dataset, config_json, status, report_json, error, created_at, completed_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (run_id, kind, dataset, json.dumps(config or {}, ensure_ascii=False),
                     "running", "", "", time.time(), None),
                )
                conn.commit()
            finally:
                conn.close()

    def finish(self, run_id: str, report: dict, status: str = "done") -> None:
        with _LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE evaluation_runs SET status=?, report_json=?, completed_at=?, error=''"
                    " WHERE id=?",
                    (status, json.dumps(report, ensure_ascii=False), time.time(), run_id),
                )
                conn.commit()
            finally:
                conn.close()

    def fail(self, run_id: str, error: str) -> None:
        with _LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE evaluation_runs SET status='failed', error=?, completed_at=? WHERE id=?",
                    (str(error)[:2000], time.time(), run_id),
                )
                conn.commit()
            finally:
                conn.close()

    def save_report(self, kind: str, dataset: str, report: dict,
                    config: dict | None = None, run_id: str | None = None) -> str:
        """一次写入完整评测记录（CLI 脚本用）：生成 run_id，直接落盘 done。"""
        rid = run_id or uuid.uuid4().hex[:12]
        with _LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO evaluation_runs"
                    "(id, kind, dataset, config_json, status, report_json, error, created_at, completed_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, kind, dataset, json.dumps(config or {}, ensure_ascii=False),
                     "done", json.dumps(report, ensure_ascii=False), "", time.time(), time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        return rid

    def get(self, run_id: str) -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id, kind, dataset, config_json, status, report_json, error, created_at, completed_at"
                " FROM evaluation_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        conn = self._conn()
        try:
            sql = ("SELECT id, kind, dataset, config_json, status, report_json, error, created_at, completed_at"
                   " FROM evaluation_runs")
            params: list = []
            if kind:
                sql += " WHERE kind=?"
                params.append(kind)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> dict:
        (rid, kind, dataset, config_json, status, report_json, error, created_at, completed_at) = row
        return {
            "id": rid,
            "kind": kind,
            "dataset": dataset,
            "config": json.loads(config_json or "{}"),
            "status": status,
            "report": json.loads(report_json) if report_json else None,
            "error": error,
            "created_at": created_at,
            "completed_at": completed_at,
        }


_STORE: EvaluationStore | None = None


def get_evaluation_store() -> EvaluationStore:
    global _STORE
    if _STORE is None:
        _STORE = EvaluationStore()
    return _STORE
