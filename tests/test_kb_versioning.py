from __future__ import annotations

import asyncio


def test_version_store_activation_failure_and_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_VERSION_DB", str(tmp_path / "versions.db"))
    from app.knowledge.versioning import DocumentVersionStore

    store = DocumentVersionStore()
    v1 = store.begin("alice", "papers", "a.pdf", "hash-1")
    store.mark_indexing(v1["version_id"])
    activated1 = store.activate(v1["version_id"], 3)
    assert activated1["revision"] == 1

    v2 = store.begin("alice", "papers", "a.pdf", "hash-2")
    store.fail(v2["version_id"], "embedding unavailable")
    assert store.revision("alice", "papers") == 1
    versions = store.versions("alice", "papers", "a.pdf")
    assert versions[0]["status"] == "failed"
    assert versions[1]["status"] == "active"

    v3 = store.begin("alice", "papers", "a.pdf", "hash-3")
    activated3 = store.activate(v3["version_id"], 4)
    assert activated3["previous_index_doc_id"] == v1["index_doc_id"]
    assert activated3["revision"] == 2
    assert store.versions("alice", "papers", "a.pdf")[0]["status"] == "active"


def test_active_content_is_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_VERSION_DB", str(tmp_path / "versions.db"))
    from app.knowledge.versioning import DocumentVersionStore

    store = DocumentVersionStore()
    pending = store.begin("alice", None, "a.pdf", "same")
    store.activate(pending["version_id"], 1)
    duplicate = store.begin("alice", None, "a.pdf", "same")
    assert duplicate["duplicate"] is True


def test_delete_deactivates_version_and_bumps_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_VERSION_DB", str(tmp_path / "versions.db"))
    from app.knowledge.versioning import DocumentVersionStore

    store = DocumentVersionStore()
    pending = store.begin("alice", None, "a.pdf", "hash")
    store.activate(pending["version_id"], 1)
    assert store.deactivate_index_document("alice", None, pending["index_doc_id"]) is True
    assert store.revision("alice", None) == 2
    assert store.versions("alice", None, "a.pdf")[0]["status"] == "deleted"


def test_user_kb_keeps_old_version_when_new_index_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_VERSION_DB", str(tmp_path / "versions.db"))
    import app.knowledge.user_kb as module

    async def no_summary(_text):
        return ""

    monkeypatch.setattr(module, "_generate_doc_summary", no_summary)
    module._KB_UPDATE_LOCKS.clear()

    class FakeStore:
        def __init__(self):
            self.docs = []

        def documents(self):
            return list(self.docs)

    class FakeRetriever:
        def __init__(self, store):
            self.store = store
            self.fail = False

        def ingest_documents(self, docs, clear=False):
            if self.fail:
                raise RuntimeError("embedding failed")
            doc = docs[0]
            self.store.docs.append({"id": doc["doc_id"] + "#0", "text": doc["text"],
                                    "metadata": {**doc["metadata"], "source": doc["source"],
                                                 "doc_id": doc["doc_id"], "chunk_id": doc["doc_id"] + "#0"}})
            return 1

        def delete_by_metadata(self, field, value):
            before = len(self.store.docs)
            self.store.docs = [d for d in self.store.docs if d["metadata"].get(field) != value]
            return len(self.store.docs) != before

    kb = object.__new__(module.UserKnowledgeBase)
    kb.username, kb.kb_name = "alice", "papers"
    kb._store = FakeStore()
    kb._retriever = FakeRetriever(kb._store)

    async def scenario():
        first = await kb.add_document("a.pdf", "old content")
        kb._retriever.fail = True
        try:
            await kb.add_document("a.pdf", "new content")
        except RuntimeError:
            pass
        else:
            raise AssertionError("index failure must propagate")
        return first

    first = asyncio.run(scenario())
    assert first["version_number"] == 1
    assert len(kb._store.docs) == 1
    assert kb._store.docs[0]["text"] == "old content"


def test_successful_update_switches_version_then_removes_old(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_VERSION_DB", str(tmp_path / "versions.db"))
    import app.knowledge.user_kb as module

    async def no_summary(_text):
        return ""

    monkeypatch.setattr(module, "_generate_doc_summary", no_summary)
    module._KB_UPDATE_LOCKS.clear()

    class FakeStore:
        def __init__(self): self.docs = []
        def documents(self): return list(self.docs)

    class FakeRetriever:
        def __init__(self, store): self.store = store
        def ingest_documents(self, docs, clear=False):
            doc = docs[0]
            self.store.docs.append({"text": doc["text"], "metadata": {
                **doc["metadata"], "source": doc["source"], "doc_id": doc["doc_id"]}})
            return 1
        def delete_by_metadata(self, field, value):
            before = len(self.store.docs)
            self.store.docs = [d for d in self.store.docs if d["metadata"].get(field) != value]
            return before != len(self.store.docs)

    kb = object.__new__(module.UserKnowledgeBase)
    kb.username, kb.kb_name, kb._store = "alice", None, FakeStore()
    kb._retriever = FakeRetriever(kb._store)

    async def scenario():
        await kb.add_document("a.pdf", "old content")
        return await kb.add_document("a.pdf", "new content")

    second = asyncio.run(scenario())
    assert second["version_number"] == 2
    assert second["kb_revision"] == 2
    assert second["version_updated"] is True
    assert [d["text"] for d in kb._store.docs] == ["new content"]
