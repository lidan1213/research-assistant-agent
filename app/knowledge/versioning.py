"""Persistent document versions and knowledge-base revisions.

SQLite is the source of truth for version state; vector/BM25 indexes remain
derived data. A new version is activated only after indexing succeeds.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()


def _db_path() -> Path:
    return Path(os.environ.get("KB_VERSION_DB", "data/document_versions.db"))


def _kb(kb_name: str | None) -> str:
    return kb_name or "__default__"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kb_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            kb_name TEXT NOT NULL,
            source TEXT NOT NULL,
            logical_doc_id TEXT NOT NULL,
            active_version_id TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(username, kb_name, source)
        );
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            index_doc_id TEXT NOT NULL,
            status TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            error_message TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            activated_at REAL,
            FOREIGN KEY(document_id) REFERENCES kb_documents(id),
            UNIQUE(document_id, version_number)
        );
        CREATE INDEX IF NOT EXISTS idx_versions_document_status
            ON document_versions(document_id, status);
        CREATE TABLE IF NOT EXISTS kb_revisions (
            username TEXT NOT NULL,
            kb_name TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            PRIMARY KEY(username, kb_name)
        );
        """
    )
    return conn


class DocumentVersionStore:
    def begin(
        self, username: str, kb_name: str | None, source: str, content_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a pending version; return duplicate=True for active content."""
        now, library = time.time(), _kb(kb_name)
        with _LOCK, _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            doc = conn.execute(
                "SELECT * FROM kb_documents WHERE username=? AND kb_name=? AND source=?",
                (username, library, source),
            ).fetchone()
            if doc is None:
                logical = f"doc_{uuid.uuid4().hex[:16]}"
                cur = conn.execute(
                    "INSERT INTO kb_documents(username,kb_name,source,logical_doc_id,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)", (username, library, source, logical, now, now),
                )
                document_id, active_version_id = cur.lastrowid, None
            else:
                document_id, logical = int(doc["id"]), str(doc["logical_doc_id"])
                active_version_id = doc["active_version_id"]
                if active_version_id:
                    active = conn.execute(
                        "SELECT content_hash FROM document_versions WHERE version_id=?",
                        (active_version_id,),
                    ).fetchone()
                    if active and active["content_hash"] == content_hash:
                        return {"duplicate": True, "logical_doc_id": logical,
                                "active_version_id": active_version_id}
            number = int(conn.execute(
                "SELECT COALESCE(MAX(version_number),0)+1 FROM document_versions WHERE document_id=?",
                (document_id,),
            ).fetchone()[0])
            version_id = f"ver_{uuid.uuid4().hex[:16]}"
            index_doc_id = f"{logical}@v{number}"
            conn.execute(
                "INSERT INTO document_versions(version_id,document_id,version_number,content_hash,index_doc_id,"
                "status,metadata_json,created_at) VALUES(?,?,?,?,?,'pending',?,?)",
                (version_id, document_id, number, content_hash, index_doc_id,
                 json.dumps(metadata or {}, ensure_ascii=False), now),
            )
            return {"duplicate": False, "version_id": version_id, "version_number": number,
                    "logical_doc_id": logical, "index_doc_id": index_doc_id,
                    "previous_version_id": active_version_id}

    def mark_indexing(self, version_id: str) -> None:
        with _LOCK, _connect() as conn:
            conn.execute("UPDATE document_versions SET status='indexing' WHERE version_id=?", (version_id,))

    def fail(self, version_id: str, error: str) -> None:
        with _LOCK, _connect() as conn:
            conn.execute("UPDATE document_versions SET status='failed',error_message=? WHERE version_id=?",
                         (error[:1000], version_id))

    def activate(self, version_id: str, chunk_count: int) -> dict[str, Any]:
        """Atomically activate version and increment the owning KB revision."""
        now = time.time()
        with _LOCK, _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT v.*,d.username,d.kb_name,d.active_version_id FROM document_versions v "
                "JOIN kb_documents d ON d.id=v.document_id WHERE v.version_id=?", (version_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"未知文档版本: {version_id}")
            previous = None
            if row["active_version_id"]:
                previous = conn.execute(
                    "SELECT version_id,index_doc_id FROM document_versions WHERE version_id=?",
                    (row["active_version_id"],),
                ).fetchone()
                conn.execute("UPDATE document_versions SET status='superseded' WHERE version_id=?",
                             (row["active_version_id"],))
            conn.execute(
                "UPDATE document_versions SET status='active',chunk_count=?,activated_at=? WHERE version_id=?",
                (chunk_count, now, version_id),
            )
            conn.execute("UPDATE kb_documents SET active_version_id=?,updated_at=? WHERE id=?",
                         (version_id, now, row["document_id"]))
            conn.execute(
                "INSERT INTO kb_revisions(username,kb_name,revision,updated_at) VALUES(?,?,1,?) "
                "ON CONFLICT(username,kb_name) DO UPDATE SET revision=revision+1,updated_at=excluded.updated_at",
                (row["username"], row["kb_name"], now),
            )
            revision = conn.execute(
                "SELECT revision FROM kb_revisions WHERE username=? AND kb_name=?",
                (row["username"], row["kb_name"]),
            ).fetchone()[0]
            return {"previous_version_id": previous["version_id"] if previous else None,
                    "previous_index_doc_id": previous["index_doc_id"] if previous else None,
                    "revision": int(revision)}

    def revision(self, username: str, kb_name: str | None) -> int:
        with _LOCK, _connect() as conn:
            row = conn.execute("SELECT revision FROM kb_revisions WHERE username=? AND kb_name=?",
                               (username, _kb(kb_name))).fetchone()
            return int(row[0]) if row else 0

    def bump_revision(self, username: str, kb_name: str | None) -> int:
        now, library = time.time(), _kb(kb_name)
        with _LOCK, _connect() as conn:
            conn.execute(
                "INSERT INTO kb_revisions(username,kb_name,revision,updated_at) VALUES(?,?,1,?) "
                "ON CONFLICT(username,kb_name) DO UPDATE SET revision=revision+1,updated_at=excluded.updated_at",
                (username, library, now),
            )
            return int(conn.execute(
                "SELECT revision FROM kb_revisions WHERE username=? AND kb_name=?",
                (username, library),
            ).fetchone()[0])

    def deactivate_index_document(self, username: str, kb_name: str | None, index_doc_id: str) -> bool:
        """Mark an explicitly deleted active version and invalidate its KB revision."""
        with _LOCK, _connect() as conn:
            row = conn.execute(
                "SELECT v.version_id,d.id FROM document_versions v JOIN kb_documents d ON d.id=v.document_id "
                "WHERE d.username=? AND d.kb_name=? AND v.index_doc_id=? AND d.active_version_id=v.version_id",
                (username, _kb(kb_name), index_doc_id),
            ).fetchone()
            if not row:
                return False
            conn.execute("UPDATE document_versions SET status='deleted' WHERE version_id=?", (row["version_id"],))
            conn.execute("UPDATE kb_documents SET active_version_id=NULL,updated_at=? WHERE id=?",
                         (time.time(), row["id"]))
        self.bump_revision(username, kb_name)
        return True

    def versions(self, username: str, kb_name: str | None, source: str) -> list[dict[str, Any]]:
        with _LOCK, _connect() as conn:
            rows = conn.execute(
                "SELECT v.version_id,v.version_number,v.content_hash,v.index_doc_id,v.status,v.chunk_count,"
                "v.error_message,v.created_at,v.activated_at FROM document_versions v JOIN kb_documents d "
                "ON d.id=v.document_id WHERE d.username=? AND d.kb_name=? AND d.source=? "
                "ORDER BY v.version_number DESC", (username, _kb(kb_name), source),
            ).fetchall()
            return [dict(row) for row in rows]


version_store = DocumentVersionStore()
