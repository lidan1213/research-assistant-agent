"""Short-term memory module boundaries and compatibility."""
from __future__ import annotations

import asyncio
import uuid

from app.llm.base import ChatMessage, MessageRole, ToolCall


def test_legacy_memory_exports_new_service_and_inmemory_store():
    from app.agent.memory import ConversationMemory as LegacyMemory
    from app.agent.memory import InMemoryStore as LegacyStore
    from app.memory.conversation import ConversationMemory
    from app.memory.stores.inmemory import InMemoryStore

    assert LegacyMemory is ConversationMemory
    assert LegacyStore is InMemoryStore


def test_conversation_memory_filters_orphan_tool_messages():
    from app.memory.conversation import ConversationMemory
    from app.memory.stores.inmemory import InMemoryStore

    async def run():
        store = InMemoryStore()
        memory = ConversationMemory(backend=store)
        await store.append("s", ChatMessage(role=MessageRole.TOOL, content="orphan", tool_call_id="x"))
        await store.append(
            "s",
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="ok", name="search", arguments="{}")],
            ),
        )
        await store.append("s", ChatMessage(role=MessageRole.TOOL, content="kept", tool_call_id="ok"))
        return await memory.history("s")

    messages = asyncio.run(run())
    assert [message.content for message in messages] == ["", "kept"]


def test_new_conversation_service_accepts_legacy_sqlite_backend():
    from app.agent.memory import SQLiteStore
    from app.memory.conversation import ConversationMemory

    path = f"var/memory-{uuid.uuid4().hex}.db"

    async def run():
        store = SQLiteStore(path)
        memory = ConversationMemory(backend=store)
        await memory.add_user("session", "question")
        await memory.add_assistant("session", "answer")
        messages = await memory.history("session")
        store.close()
        return messages

    assert [message.content for message in asyncio.run(run())] == ["question", "answer"]
