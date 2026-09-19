"""Conversation message-store contract."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.llm.base import ChatMessage


class BaseStore(ABC):
    @abstractmethod
    async def append(self, session_id: str, msg: ChatMessage) -> None: ...

    @abstractmethod
    async def get(self, session_id: str) -> list[ChatMessage]: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...

