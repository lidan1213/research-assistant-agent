"""Vector-store package boundaries and backend selection."""
from __future__ import annotations

import pytest
import uuid


def test_legacy_and_new_faiss_exports_are_identical():
    from app.knowledge.stores.faiss import FAISSVectorStore as NewExport
    from app.knowledge.vectorstore import FAISSVectorStore as LegacyExport

    assert NewExport is LegacyExport


def test_factory_rejects_unknown_backend():
    from app.knowledge.stores import create_vector_store

    with pytest.raises(ValueError, match="Unknown vector-store backend"):
        create_vector_store("typo-backend")


def test_legacy_factory_is_new_factory_export():
    from app.knowledge.stores.factory import create_vector_store as new_factory
    from app.knowledge.vectorstore import create_vector_store as legacy_factory

    assert legacy_factory is new_factory


def test_http_chroma_exposes_public_cache_refresh():
    from app.knowledge.stores.chroma_http import RestChromaVectorStore

    assert callable(RestChromaVectorStore.refresh_cache)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class _FakeChromaClient:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/v1/collections") and method == "GET":
            return _FakeResponse([{"id": "cid", "name": "unit"}])
        if url.endswith("/cid/get"):
            return _FakeResponse({"ids": [], "documents": [], "metadatas": []})
        if url.endswith("/cid/query"):
            return _FakeResponse(
                {
                    "ids": [["d1#0"]],
                    "distances": [[0.2]],
                    "documents": [["document"]],
                    "metadatas": [[{"doc_id": "d1"}]],
                }
            )
        if url.endswith("/cid/add"):
            return _FakeResponse({})
        if url.endswith("/api/v1/collections") and method == "POST":
            return _FakeResponse({"id": "new-cid"})
        return _FakeResponse({})


def test_http_chroma_works_with_injected_client():
    from app.knowledge.stores.chroma_http import RestChromaVectorStore

    client = _FakeChromaClient()
    store = RestChromaVectorStore(
        "http://chroma.test", collection_name="unit", client=client
    )
    store.add([[1.0, 0.0]], [{"text": "document", "metadata": {"chunk_id": "d1#0"}}])
    results = store.search([1.0, 0.0], 1)

    assert store.count() == 1
    assert results[0]["score"] == pytest.approx(0.8)
    assert results[0]["metadata"]["doc_id"] == "d1"


def test_all_legacy_store_exports_point_to_new_modules():
    from app.knowledge.stores.chroma_http import RestChromaVectorStore as HttpNew
    from app.knowledge.stores.chroma_local import ChromaVectorStore as LocalNew
    from app.knowledge.vectorstore import ChromaVectorStore as LocalLegacy
    from app.knowledge.vectorstore import RestChromaVectorStore as HttpLegacy

    assert HttpLegacy is HttpNew
    assert LocalLegacy is LocalNew


def test_sqlite_persists_and_replaces_by_chunk_id():
    from app.knowledge.stores.sqlite import SQLiteVectorStore

    path = f"var/test-{uuid.uuid4().hex}.db"
    store = SQLiteVectorStore(str(path))
    store.add([[1.0, 0.0]], [{"text": "old", "metadata": {"chunk_id": "d1#0"}}])
    store.add([[0.0, 1.0]], [{"text": "new", "metadata": {"chunk_id": "d1#0"}}])

    assert store.count() == 1
    assert store.search([0.0, 1.0], 1)[0]["text"] == "new"
    reloaded = SQLiteVectorStore(str(path))
    assert reloaded.count() == 1
    assert reloaded.search([0.0, 1.0], 1)[0]["id"] == "d1#0"


def test_faiss_persists_when_dependency_available():
    pytest.importorskip("faiss")
    from app.knowledge.stores.faiss import FAISSVectorStore

    path = f"var/test-{uuid.uuid4().hex}.faiss"
    store = FAISSVectorStore(str(path))
    store.add([[1.0, 0.0]], [{"text": "document", "metadata": {"doc_id": "d1"}}])

    assert store.count() == 1
    reloaded = FAISSVectorStore(str(path))
    assert reloaded.count() == 1
    assert reloaded.search([1.0, 0.0], 1)[0]["metadata"]["doc_id"] == "d1"
