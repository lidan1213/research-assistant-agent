"""FAISS vector-store backend."""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("vectorstore.faiss")


class FAISSVectorStore:
    def __init__(self, index_path: str | None = None) -> None:
        self.index_path = index_path or os.path.join(
            get_settings().knowledge.knowledge_dir, "index.faiss"
        )
        self.meta_path = self.index_path + ".meta.json"
        self._docs: list[dict] = []
        self._index = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            import faiss

            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                self._index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as file:
                    self._docs = json.load(file)
                logger.info(f"已加载向量库，文档数={len(self._docs)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"向量库加载失败（将新建空库）: {exc}")

    def _persist(self) -> None:
        try:
            import faiss

            parent = os.path.dirname(self.index_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as file:
                json.dump(self._docs, file, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"向量库持久化失败: {exc}")

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        if not embeddings:
            return
        array = np.asarray(embeddings, dtype="float32")
        with self._lock:
            if self._index is None:
                import faiss

                self._index = faiss.IndexFlatIP(array.shape[1])
            start = len(self._docs)
            self._index.add(array)
            for index, document in enumerate(docs):
                self._docs.append({"id": start + index, **document})
        self._persist()

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        with self._lock:
            query = np.asarray([query_vec], dtype="float32")
            scores, indexes = self._index.search(query, min(top_k, self._index.ntotal))
        results = []
        for score, index in zip(scores[0], indexes[0]):
            if index != -1:
                results.append({"score": float(score), **self._docs[index]})
        return results

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs = []
        self._index = None
        for path in (self.index_path, self.meta_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning(f"向量库文件删除失败（已重置内存索引）: {path} ({exc})")


__all__ = ["FAISSVectorStore"]
