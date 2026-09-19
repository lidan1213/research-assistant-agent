"""Conversation-memory service independent of concrete persistence backends."""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, MessageRole
from app.memory.stores.base import BaseStore
from app.memory.stores.inmemory import InMemoryStore

logger = get_logger("conversation_memory")


class ConversationMemory:
    """Unified conversation API with message and token-based sliding windows."""

    def __init__(self, window: int = 20, backend: BaseStore | None = None) -> None:
        self.window = window
        self.token_budget = 0
        if backend is not None:
            self._store = backend
            return

        settings = get_settings()
        if settings.memory.redis_url:
            from app.memory.stores.redis import RedisStore

            logger.info("使用 Redis 作为记忆后端（双写 SQLite 持久化兜底）")
            self._store = RedisStore(
                settings.memory.redis_url,
                settings.memory.ttl,
                persist_path=settings.memory.sqlite_path,
            )
        elif settings.memory.backend == "sqlite":
            from app.memory.stores.sqlite import SQLiteStore

            logger.info("使用 SQLite 作为会话记忆后端（落盘持久化）")
            self._store = SQLiteStore(settings.memory.sqlite_path)
        else:
            self._store = InMemoryStore()

    async def add_user(self, session_id: str, text: str) -> None:
        await self._store.append(
            session_id, ChatMessage(role=MessageRole.USER, content=text)
        )

    @staticmethod
    async def _maybe_await(result):
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def get_title(self, session_id: str) -> str | None:
        if hasattr(self._store, "get_title"):
            return await self._maybe_await(self._store.get_title(session_id))
        return None

    async def set_title(self, session_id: str, title: str) -> None:
        if hasattr(self._store, "set_title"):
            await self._maybe_await(self._store.set_title(session_id, title))

    async def set_summary(self, session_id: str, summary: str) -> None:
        if hasattr(self._store, "set_summary"):
            await self._maybe_await(self._store.set_summary(session_id, summary))

    async def get_summary(self, session_id: str) -> str | None:
        if hasattr(self._store, "get_summary"):
            return await self._maybe_await(self._store.get_summary(session_id))
        return None

    async def add_assistant(
        self, session_id: str, text: str, tool_calls: list | None = None
    ) -> None:
        await self._store.append(
            session_id,
            ChatMessage(
                role=MessageRole.ASSISTANT, content=text, tool_calls=tool_calls
            ),
        )

    async def add_tool(
        self, session_id: str, tool_call_id: str, name: str, content: str
    ) -> None:
        await self._store.append(
            session_id,
            ChatMessage(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            ),
        )

    async def history(self, session_id: str) -> list[ChatMessage]:
        messages = await self._store.get(session_id)
        if self.token_budget > 0:
            messages = messages[self._truncate_by_tokens(messages) :]
        elif len(messages) > self.window:
            start = len(messages) - self.window
            while start > 0 and messages[start].role == MessageRole.TOOL:
                start -= 1
            messages = messages[start:]

        cleaned = []
        pending_ids: set[str] = set()
        for message in messages:
            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                pending_ids = {call.id for call in message.tool_calls}
                cleaned.append(message)
            elif message.role == MessageRole.TOOL:
                if message.tool_call_id in pending_ids:
                    cleaned.append(message)
            else:
                pending_ids = set()
                cleaned.append(message)
        return cleaned

    @staticmethod
    def _approx_tokens(message: ChatMessage) -> int:
        tool_extra = sum(
            len(call.arguments or "") // 2 + 20 for call in (message.tool_calls or [])
        )
        return len(message.content or "") // 2 + tool_extra + 4

    def _truncate_by_tokens(self, messages: list[ChatMessage]) -> int:
        if not messages:
            return 0
        total = 0
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            total += self._approx_tokens(messages[index])
            if total > self.token_budget:
                start = index + 1
                break
        while start > 0 and start < len(messages) and messages[start].role == MessageRole.TOOL:
            start -= 1
        if start >= len(messages) or all(
            message.role == MessageRole.TOOL for message in messages[start:]
        ):
            start = max(0, len(messages) - self.window)
            while start > 0 and messages[start].role == MessageRole.TOOL:
                start -= 1
        return start

    async def clear(self, session_id: str) -> None:
        await self._store.clear(session_id)
