"""Backward-compatible vector-store imports.

New code should import from :mod:`app.knowledge.stores`.  This module keeps the
historical public names available while concrete implementations live in
backend-specific modules.
"""
from __future__ import annotations

from app.knowledge.stores.chroma_http import RestChromaVectorStore
from app.knowledge.stores.chroma_local import ChromaVectorStore
from app.knowledge.stores.factory import create_vector_store
from app.knowledge.stores.faiss import FAISSVectorStore
from app.knowledge.stores.sqlite import SQLiteVectorStore

__all__ = [
    "ChromaVectorStore",
    "FAISSVectorStore",
    "RestChromaVectorStore",
    "SQLiteVectorStore",
    "create_vector_store",
]
