"""Vector-store backend selection."""
from __future__ import annotations

from app.config import get_settings
from app.knowledge.stores.base import VectorStore


def create_vector_store(backend: str | None = None) -> VectorStore:
    """Create the configured backend while keeping optional imports lazy."""
    selected = (backend or get_settings().knowledge.vector_store).strip().lower()
    if selected == "chroma":
        from app.knowledge.stores.chroma_http import RestChromaVectorStore

        return RestChromaVectorStore()
    if selected == "chroma-local":
        from app.knowledge.stores.chroma_local import ChromaVectorStore

        return ChromaVectorStore()
    if selected == "sqlite":
        from app.knowledge.stores.sqlite import SQLiteVectorStore

        return SQLiteVectorStore()
    if selected == "faiss":
        from app.knowledge.stores.faiss import FAISSVectorStore

        return FAISSVectorStore()
    raise ValueError(
        f"Unknown vector-store backend {selected!r}; expected faiss, sqlite, chroma, or chroma-local"
    )
