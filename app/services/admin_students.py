"""Administrator operations for student accounts and sessions."""
from __future__ import annotations

from app.config import get_settings
from app.core.auth import ROLE_STUDENT, UserStore
from app.core.exceptions import ForbiddenError
from app.memory.stores.sqlite import SQLiteStore


class AdminStudentService:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or get_settings().memory.sqlite_path

    @staticmethod
    def _require_student(username: str):
        user = UserStore().get(username)
        if user is None or user.role != ROLE_STUDENT:
            raise ForbiddenError(f"用户 {username} 不是学生账号")
        return user

    def list_students(self) -> list[dict]:
        store = SQLiteStore(self.db_path)
        try:
            return [
                {
                    "username": user.username,
                    "session_count": len(store.list_sessions(prefix=f"{user.username}__")),
                }
                for user in UserStore().list_by_role(ROLE_STUDENT)
            ]
        finally:
            store.close()

    def reset_password(self, username: str, password: str) -> None:
        self._require_student(username)
        if not UserStore().reset_password(username, password):
            raise ForbiddenError(f"用户 {username} 不存在")

    def list_sessions(self, username: str) -> list[dict]:
        self._require_student(username)
        store = SQLiteStore(self.db_path)
        try:
            return store.list_sessions(prefix=f"{username}__")
        finally:
            store.close()

    async def session_detail(self, username: str, session_id: str) -> list[dict]:
        self._require_student(username)
        if not session_id.startswith(f"{username}__"):
            raise ForbiddenError("会话不属于该学生")
        store = SQLiteStore(self.db_path)
        try:
            return [message.to_dict() for message in await store.get(session_id)]
        finally:
            store.close()


admin_student_service = AdminStudentService()
