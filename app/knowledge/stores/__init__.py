"""Vector-store backends and factory."""

from app.knowledge.stores.base import VectorStore
from app.knowledge.stores.factory import create_vector_store

__all__ = ["VectorStore", "create_vector_store"]
