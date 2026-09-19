"""Stable retrieval contract exposed to application boundaries."""
from __future__ import annotations

from typing import Protocol


class RetrievalService(Protocol):
    def ingest(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        ...

    def ingest_documents(self, docs: list[dict], *, clear: bool = False) -> int:
        ...

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        ...

    def count(self) -> int:
        ...

    def format_context(self, candidates: list[dict]) -> str:
        ...
