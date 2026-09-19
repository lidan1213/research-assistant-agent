"""Events emitted by agent runtimes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: str
    data: dict

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}

