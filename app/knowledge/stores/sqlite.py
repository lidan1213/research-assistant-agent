"""SQLite vector-store backend."""
from __future__ import annotations

import json
import os
import sqlite3
import threading

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("vectorstore.sqlite")


class SQLiteVectorStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.path.join(
            get_settings().knowledge.knowledge_dir, "knowledge.db"
        )
        self.index_path = self.db_path
        self.meta_path = self.db_path + ".meta.json"
        self._docs: list[dict] = []
        self._lock = threading.RLock()
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE,
                    doc_id TEXT,
                    text TEXT,
                    metadata_json TEXT,
                    vector BLOB
                )"""
            )
            self._conn.commit()
        self._load()

    def _load(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, text, metadata_json FROM chunks ORDER BY id"
            ).fetchall()
        self._docs = [
            {"id": chunk_id, "text": text, "metadata": json.loads(metadata or "{}")}
            for chunk_id, text, metadata in rows
        ]
        logger.info(f"已加载 SQLite 向量库，文档数={len(self._docs)} -> {self.db_path}")

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        if not embeddings:
            return
        with self._lock:
            for index, (embedding, document) in enumerate(zip(embeddings, docs)):
                metadata = document.get("metadata") or {}
                chunk_id = metadata.get("chunk_id") or f"id_{len(self._docs) + index}"
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, metadata_json, vector) "
                    "VALUES (?,?,?,?,?)",
                    (
                        str(chunk_id),
                        metadata.get("doc_id", ""),
                        document.get("text", ""),
                        json.dumps(metadata, ensure_ascii=False),
                        np.asarray(embedding, dtype="float32").tobytes(),
                    ),
                )
            self._conn.commit()
        self._load()

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, text, metadata_json, vector FROM chunks"
            ).fetchall()
        if not rows:
            return []
        query = np.asarray(query_vec, dtype="float32")
        vectors = np.stack([np.frombuffer(row[3], dtype="float32") for row in rows])
        scores = vectors @ query
        indexes = np.argsort(-scores)[:top_k]
        return [
            {
                "score": float(scores[index]),
                "id": rows[index][0],
                "text": rows[index][1],
                "metadata": json.loads(rows[index][2] or "{}"),
            }
            for index in indexes
        ]

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.commit()
        self._docs = []


__all__ = ["SQLiteVectorStore"]
