"""Request-scoped active Skill context shared by all Agent runtimes."""
from __future__ import annotations

from contextvars import ContextVar

active_skill_prompt: ContextVar[str] = ContextVar("active_skill_prompt", default="")


def append_active_skill(base: str) -> str:
    prompt = active_skill_prompt.get().strip()
    if not prompt:
        return base
    return f"{base.rstrip()}\n\n{prompt}"
