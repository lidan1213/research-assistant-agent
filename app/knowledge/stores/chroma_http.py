"""Remote ChromaDB backend with a shared HTTP connection pool and caches."""
from __future__ import annotations

import os
import threading
import time

import httpx

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("vectorstore.chroma_http")

_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()
_COLLECTION_CACHE: dict[tuple[str, str], tuple[str | None, float]] = {}
_DOCS_CACHE: dict[tuple[str, str], tuple[list, float]] = {}
_CACHE_TTL = 30.0
_FAILURE_CACHE_TTL = 60.0
_CACHE_LOCK = threading.Lock()


def get_shared_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(
                    timeout=httpx.Timeout(3.0, connect=1.0),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _CLIENT


def _key(base_url: str, collection_name: str) -> tuple[str, str]:
    return base_url, collection_name


def get_collection_id(
    base_url: str, name: str, client: httpx.Client
) -> str | None:
    key = _key(base_url, name)
    now = time.time()
    with _CACHE_LOCK:
        cached = _COLLECTION_CACHE.get(key)
        ttl = _CACHE_TTL if cached and cached[0] else _FAILURE_CACHE_TTL
        if cached and now - cached[1] < ttl:
            return cached[0]
    try:
        collections = client.request("GET", f"{base_url}/api/v1/collections").json()
        collection_id = next(
            (item["id"] for item in collections if item.get("name") == name), None
        )
        if not collection_id:
            collection_id = client.request(
                "POST",
                f"{base_url}/api/v1/collections",
                json={"name": name, "metadata": {"hnsw:space": "cosine"}},
            ).json()["id"]
        with _CACHE_LOCK:
            _COLLECTION_CACHE[key] = collection_id, time.time()
        return collection_id
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ChromaDB 服务端连接失败（collection={name}）: {exc}")
        # 负缓存：故障期内不让每个请求都重复等待连接超时。
        with _CACHE_LOCK:
            _COLLECTION_CACHE[key] = None, time.time()
        return None


def get_docs(
    base_url: str, name: str, collection_id: str, client: httpx.Client
) -> list:
    key = _key(base_url, name)
    now = time.time()
    with _CACHE_LOCK:
        cached = _DOCS_CACHE.get(key)
        if cached and now - cached[1] < _CACHE_TTL:
            return list(cached[0])
    docs = []
    try:
        result = client.request(
            "POST", f"{base_url}/api/v1/collections/{collection_id}/get", json={}
        ).json()
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        docs = [
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ChromaDB 全量拉取失败（collection={name}）: {exc}")
    set_docs(base_url, name, docs)
    return docs


def set_docs(base_url: str, name: str, docs: list) -> None:
    with _CACHE_LOCK:
        _DOCS_CACHE[_key(base_url, name)] = list(docs), time.time()


def invalidate_cache(base_url: str, name: str) -> None:
    with _CACHE_LOCK:
        _DOCS_CACHE.pop(_key(base_url, name), None)
        _COLLECTION_CACHE.pop(_key(base_url, name), None)


class RestChromaVectorStore:
    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str = "knowledge",
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001")
        ).rstrip("/")
        self.collection_name = collection_name
        self.index_path = self.base_url
        self.meta_path = os.path.join(
            get_settings().knowledge.knowledge_dir, "chroma_rest.meta.json"
        )
        self._docs: list[dict] = []
        self._client = client or get_shared_client()
        self._collection_id = get_collection_id(
            self.base_url, self.collection_name, self._client
        )
        if self._collection_id:
            self._docs = get_docs(
                self.base_url, self.collection_name, self._collection_id, self._client
            )

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _sanitize_meta(metadata: dict) -> dict:
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        if not embeddings or not self._collection_id:
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
            self._request(
                "POST",
                f"/api/v1/collections/{self._collection_id}/add",
                json={
                    "ids": ids,
                    "embeddings": [list(item) for item in embeddings],
                    "documents": documents,
                    "metadatas": metadatas,
                },
            )
            self._docs.extend({"id": ids[index], **doc} for index, doc in enumerate(docs))
            self.refresh_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 写入失败（跳过本批）: {exc}")

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        if not self._collection_id:
            return []
        try:
            result = self._request(
                "POST",
                f"/api/v1/collections/{self._collection_id}/query",
                json={"query_embeddings": [list(query_vec)], "n_results": max(top_k, 1)},
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
                for index, item_id in enumerate(ids[:top_k])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 检索失败（返回空）: {exc}")
            return []

    def count(self) -> int:
        return len(self._docs)

    def documents(self) -> list[dict]:
        """Return a snapshot of indexed document metadata."""
        return list(self._docs)

    def delete_by_metadata(self, field: str, value: str) -> bool:
        """Delete documents matching one metadata field and refresh local caches."""
        if not self._collection_id:
            return False
        try:
            self._request(
                "POST",
                f"/api/v1/collections/{self._collection_id}/delete",
                json={"where": {field: {"$eq": value}}},
            )
            self._docs = [
                document
                for document in self._docs
                if (document.get("metadata") or {}).get(field) != value
            ]
            self.refresh_cache()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB 按元数据删除失败: %s", exc)
            return False

    def refresh_cache(self) -> None:
        set_docs(self.base_url, self.collection_name, self._docs)

    def clear(self) -> None:
        self._docs = []
        invalidate_cache(self.base_url, self.collection_name)
        try:
            if self._collection_id:
                self._request("DELETE", f"/api/v1/collections/{self.collection_name}")
            created = self._request(
                "POST",
                "/api/v1/collections",
                json={
                    "name": self.collection_name,
                    "metadata": {"hnsw:space": "cosine"},
                },
            )
            self._collection_id = created["id"]
            with _CACHE_LOCK:
                _COLLECTION_CACHE[_key(self.base_url, self.collection_name)] = (
                    self._collection_id,
                    time.time(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ChromaDB 清空失败（已重置内存）: {exc}")


__all__ = ["RestChromaVectorStore"]
