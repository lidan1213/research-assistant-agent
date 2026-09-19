"""Process-local conversation message store."""
from __future__ import annotations

from app.llm.base import ChatMessage
from app.memory.stores.base import BaseStore


class InMemoryStore(BaseStore):
    def __init__(self) -> None:
        self._data: dict[str, list[ChatMessage]] = {}

    async def append(self, session_id: str, msg: ChatMessage) -> None:
        self._data.setdefault(session_id, []).append(msg)

    async def get(self, session_id: str) -> list[ChatMessage]:
        return list(self._data.get(session_id, []))

    async def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

