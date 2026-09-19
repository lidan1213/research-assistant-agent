"""Agent runtime contracts used at application boundaries."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.agent.events import AgentEvent


class AgentRuntime(Protocol):
    """Stable contract consumed by API routes.

    Concrete orchestration implementations may change without coupling the HTTP
    layer to their internal planner, memory, tool, or graph implementation.
    """

    async def run(self, session_id: str, goal: str, use_plan: bool = True) -> str:
        ...

    def stream(
        self, session_id: str, goal: str, use_plan: bool = True
    ) -> AsyncIterator[AgentEvent]:
        ...
