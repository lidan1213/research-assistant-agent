"""Common contract for vector-store backends."""
from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    """Minimal storage contract required by the retrieval pipeline."""

    _docs: list[dict]

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        ...

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        ...

    def clear(self) -> None:
        ...

