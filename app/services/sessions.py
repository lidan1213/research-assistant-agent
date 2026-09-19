"""User-scoped session management service."""
from __future__ import annotations

from app.config import get_settings
from app.core.audit import record_unauthorized
from app.core.exceptions import ForbiddenError
from app.memory.stores.sqlite import SQLiteStore


class SessionService:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or get_settings().memory.sqlite_path

    @staticmethod
    def prefix(username: str) -> str:
        return f"{username}__"

    def full_id(self, username: str, session_id: str) -> str:
        prefix = self.prefix(username)
        return session_id if session_id.startswith(prefix) else f"{prefix}{session_id}"

    def list(self, username: str) -> list[dict]:
        prefix = self.prefix(username)
        store = SQLiteStore(self.db_path)
        try:
            rows = store.list_sessions(prefix=prefix)
        finally:
            store.close()
        return [
            {
                **row,
                "session_id": row["session_id"][len(prefix) :],
                "title": row.get("title") or "",
                "summary": row.get("summary") or "",
                "archived": row.get("archived", False),
            }
            for row in rows
        ]

    def _require_owned(self, store: SQLiteStore, username: str, session_id: str) -> str:
        prefix = self.prefix(username)
        full_id = self.full_id(username, session_id)
        owned = {row["session_id"] for row in store.list_sessions(prefix=prefix)}
        if full_id not in owned:
            record_unauthorized(username, "session_access", session_id)
            raise ForbiddenError("会话不存在或不属于当前用户")
        return full_id

    async def history(self, username: str, session_id: str, *, require_prefix: bool = False):
        if require_prefix and not session_id.startswith(self.prefix(username)):
            raise ForbiddenError("只能查看自己的会话记录")
        store = SQLiteStore(self.db_path)
        try:
            full_id = (
                self._require_owned(store, username, session_id)
                if not require_prefix
                else session_id
            )
            return full_id, await store.get(full_id)
        finally:
            store.close()

    def rename(self, username: str, session_id: str, title: str) -> str:
        clean_title = title.strip()[:50]
        if not clean_title:
            raise ForbiddenError("标题不能为空")
        store = SQLiteStore(self.db_path)
        try:
            full_id = self._require_owned(store, username, session_id)
            store.set_title(full_id, clean_title)
        finally:
            store.close()
        return clean_title

    def restore(self, username: str, session_id: str) -> int:
        store = SQLiteStore(self.db_path)
        try:
            full_id = self._require_owned(store, username, session_id)
            store.set_archived(full_id, False)
            return 1
        finally:
            store.close()

    def retry(self, username: str, session_id: str) -> int:
        store = SQLiteStore(self.db_path)
        try:
            full_id = self._require_owned(store, username, session_id)
            return store.truncate_to_last_user(full_id)
        finally:
            store.close()

    def batch_delete(self, username: str, session_ids: list[str]) -> int:
        prefix = self.prefix(username)
        store = SQLiteStore(self.db_path)
        try:
            owned = {row["session_id"] for row in store.list_sessions(prefix=prefix)}
            deleted = 0
            for session_id in session_ids:
                full_id = self.full_id(username, session_id)
                if full_id in owned:
                    store.delete_session(full_id)
                    deleted += 1
            return deleted
        finally:
            store.close()

    def search(self, query: str, username: str, *, search_all: bool = False) -> list[dict]:
        store = SQLiteStore(self.db_path)
        try:
            prefix = "" if search_all else self.prefix(username)
            return store.search_messages(query.strip(), prefix=prefix, limit=30)
        finally:
            store.close()

    async def export_data(self, username: str, session_id: str) -> dict:
        store = SQLiteStore(self.db_path)
        try:
            full_id = self._require_owned(store, username, session_id)
            return {
                "session_id": full_id,
                "messages": await store.get(full_id),
                "title": store.get_title(full_id) or "科研助手会话",
                "summary": store.get_summary(full_id) or "",
            }
        finally:
            store.close()

    async def export_all_data(self, username: str) -> list[dict]:
        prefix = self.prefix(username)
        store = SQLiteStore(self.db_path)
        try:
            output = []
            for session in store.list_sessions(prefix=prefix):
                if session.get("archived"):
                    continue
                output.append(
                    {**session, "messages": await store.get(session["session_id"])}
                )
            return output
        finally:
            store.close()


session_service = SessionService()
