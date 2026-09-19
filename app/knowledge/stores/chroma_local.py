"""Local persistent Chroma vector-store backend."""
from __future__ import annotations

import os

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("vectorstore.chroma_local")


class ChromaVectorStore:
    def __init__(
        self, persist_dir: str | None = None, collection_name: str = "knowledge"
    ) -> None:
        self.persist_dir = persist_dir or os.path.join(
            get_settings().knowledge.knowledge_dir, "chroma"
        )
        self.index_path = self.persist_dir
        self.meta_path = os.path.join(self.persist_dir, "meta.json")
        self.collection_name = collection_name
        self._docs: list[dict] = []
        self._client = None
        self._col = None
        self._init()

    def _init(self) -> None:
        try:
            import chromadb

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._col = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            result = self._col.get(include=["documents", "metadatas"])
            ids = result.get("ids", [])
            documents = result.get("documents", []) or []
            metadatas = result.get("metadatas", []) or []
            self._docs = [
                {
                    "id": item_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": (
                        metadatas[index]
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                }
                for index, item_id in enumerate(ids)
            ]
            logger.info(f"已加载 ChromaDB 向量库，文档数={len(self._docs)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 初始化失败（降级为空库）: {exc}")
            self._col = None

    @staticmethod
    def _sanitize_meta(metadata: dict) -> dict:
        return {
            key: value
            for key, value in (metadata or {}).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        if not embeddings or self._col is None:
            return
        try:
            ids = []
            documents = []
            metadatas = []
            for index, document in enumerate(docs):
                metadata = document.get("metadata") or {}
                ids.append(str(metadata.get("chunk_id") or f"id_{len(self._docs) + index}"))
                documents.append(document.get("text", ""))
                metadatas.append(self._sanitize_meta(metadata))
            self._col.add(
                ids=ids,
                embeddings=[list(embedding) for embedding in embeddings],
                documents=documents,
                metadatas=metadatas,
            )
            self._docs.extend({"id": ids[index], **doc} for index, doc in enumerate(docs))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 写入失败（跳过本批）: {exc}")

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        if self._col is None:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []
            result = self._col.query(
                query_embeddings=[list(query_vec)],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            return [
                {
                    "score": 1.0 - float(distances[index]) if index < len(distances) else 0.0,
                    "id": item_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": (
                        metadatas[index]
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                }
                for index, item_id in enumerate(ids)
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 检索失败（返回空）: {exc}")
            return []

    def count(self) -> int:
        try:
            return self._col.count() if self._col is not None else 0
        except Exception:  # noqa: BLE001
            return len(self._docs)

    def clear(self) -> None:
        self._docs = []
        try:
            if self._client is not None and self._col is not None:
                self._client.delete_collection(self.collection_name)
                self._col = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 清空失败（已重置内存）: {exc}")


__all__ = ["ChromaVectorStore"]
