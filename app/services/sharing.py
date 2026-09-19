"""Signed, read-only session sharing."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time

from app.config import get_settings
from app.core.exceptions import ForbiddenError
from app.memory.stores.sqlite import SQLiteStore
from app.services.sessions import SessionService

SHARE_TTL = 7 * 24 * 3600


class SessionSharingService:
    def __init__(self, sessions: SessionService) -> None:
        self.sessions = sessions

    @staticmethod
    def _encode(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")

    @staticmethod
    def _decode(value: str) -> str:
        return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode()).decode()

    @staticmethod
    def _sign(payload: str) -> str:
        return hmac.new(
            get_settings().auth_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()[:24]

    def create(self, username: str, session_id: str) -> str:
        store = SQLiteStore(self.sessions.db_path)
        try:
            full_id = self.sessions._require_owned(store, username, session_id)
        finally:
            store.close()
        payload = f"{self._encode(full_id)}.{int(time.time()) + SHARE_TTL}"
        return f"{payload}.{self._sign(payload)}"

    async def read(self, share_id: str) -> dict:
        try:
            payload, signature = share_id.rsplit(".", 1)
            if not hmac.compare_digest(signature, self._sign(payload)):
                raise ForbiddenError("分享链接无效")
            encoded_id, expires_at = payload.split(".")
            if int(expires_at) < time.time():
                raise ForbiddenError("分享链接已过期")
            full_id = self._decode(encoded_id)
        except ForbiddenError:
            raise
        except Exception:  # noqa: BLE001
            raise ForbiddenError("分享链接无效") from None

        store = SQLiteStore(self.sessions.db_path)
        try:
            return {
                "messages": await store.get(full_id),
                "title": store.get_title(full_id) or "科研助手会话",
                "summary": store.get_summary(full_id) or "",
            }
        finally:
            store.close()


sharing_service = SessionSharingService(SessionService())
