"""Plan-Execute 暂停任务持久化：等待用户补充后从原步骤恢复。"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class PendingTaskStore:
    def __init__(self, path: str = "./data/pending_tasks.db", ttl_seconds: int = 86400) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS pending_tasks (
                    token TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )

    def save(self, payload: dict) -> str:
        token = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_tasks WHERE created_at < ?", (now - self.ttl_seconds,))
            conn.execute(
                "INSERT INTO pending_tasks(token, payload, created_at) VALUES (?, ?, ?)",
                (token, json.dumps(payload, ensure_ascii=False), now),
            )
        return token

    def pop(self, token: str) -> dict | None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM pending_tasks WHERE token = ?", (token,)
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM pending_tasks WHERE token = ?", (token,))
        if now - float(row["created_at"]) > self.ttl_seconds:
            return None
        return json.loads(row["payload"])


_store: PendingTaskStore | None = None


def get_pending_task_store() -> PendingTaskStore:
    global _store
    if _store is None:
        _store = PendingTaskStore()
    return _store

