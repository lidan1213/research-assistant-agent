"""嵌入模型封装。

默认使用 sentence-transformers（本地推理，无需联网 API）。
首次使用会下载模型权重，请确保网络可用或提前缓存。

也可通过配置切换为本地 Ollama 服务生成的嵌入：
- 在 .env 中设置 KNOWLEDGE__EMBEDDING_MODEL=ollama://nomic-embed-text
  （可选携带 host：ollama://localhost:11434/nomic-embed-text）
- 需本地已运行 `ollama serve` 并已拉取对应模型（如 `ollama pull nomic-embed-text`）
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.request
from typing import List

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("embedding")


class OllamaEmbedding:
    """通过本地 Ollama 服务生成嵌入（POST /api/embeddings）。

    配置方式：embedding_model = "ollama://<模型名>"，
    可选携带 host："ollama://<host:port>/<模型名>"（默认 http://localhost:11434）。
    """

    def __init__(self, spec: str, timeout: int = 60) -> None:
        rest = spec[len("ollama://"):]
        if "/" in rest:
            host_part, model = rest.rsplit("/", 1)
            if not host_part.startswith("http"):
                host_part = "http://" + host_part
            self.host = host_part
        else:
            model = rest
            self.host = "http://localhost:11434"
        self.model = model
        self.timeout = timeout

    def _call(self, payload: dict) -> dict:
        url = self.host.rstrip("/") + "/api/embeddings"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 优先批量 input（较新 Ollama 支持），失败回退逐条 prompt
        try:
            out = self._call({"model": self.model, "input": texts})
            if "embeddings" in out:
                return [list(e) for e in out["embeddings"]]
        except Exception:  # noqa: BLE001
            pass
        return [self._call({"model": self.model, "prompt": t})["embedding"] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return list(self._call({"model": self.model, "prompt": text})["embedding"])


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().knowledge.embedding_model
        # 配置以 ollama:// 开头时走本地 Ollama 服务；否则本地 sentence-transformers
        self._ollama = (
            OllamaEmbedding(self.model_name)
            if self.model_name.startswith("ollama://")
            else None
        )
        self._model = None
        self._hashing = self.model_name.startswith("hashing://")
        self._hash_model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info(f"加载嵌入模型: {self.model_name}")
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._hashing:
            if self._hash_model is None:
                dim = int(self.model_name.removeprefix("hashing://") or 512)
                self._hash_model = HashingEmbedding(dim=dim)
            return self._hash_model.embed(texts)
        if self._ollama is not None:
            return self._ollama.embed(texts)
        model = self._ensure()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._hashing:
            return self.embed([text])[0]
        if self._ollama is not None:
            return self._ollama.embed_query(text)
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# 轻量可本地运行的嵌入（无需 torch / sentence-transformers）
# 仅用于测试、本地演示与 CI；不具语义泛化能力。生产请使用 EmbeddingModel。
# ---------------------------------------------------------------------------
def _embedding_tokens(text: str) -> List[str]:
    """中文字符级 + 英文/数字词级的 token 切分，便于无模型下的词面召回。"""
    t = (text or "").lower()
    tokens: List[str] = list(re.findall(r"[一-鿿]", t))  # 中文字符级
    tokens += re.findall(r"[a-z0-9]+", t)  # 英文/数字词级
    return tokens


class HashingEmbedding:
    """基于 hashing trick 的确定性 CPU 嵌入。

    - 不依赖任何深度学习框架，毫秒级、可复现。
    - 中文字符级 + 英文词级 token 经哈希映射到定长向量后 L2 归一化，
      因而具备「词面相似度」召回能力（与 sentence-transformers 的语义召回不同）。
    - 用于：单元测试、CI、以及不装 torch 环境里的本地知识库演示。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype="float32")
        for tok in _embedding_tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            vec[idx] += 1.0 if (h & 1) == 0 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


def create_embedding(
    kind: str = "sentence-transformers", model_name: str | None = None
) -> "EmbeddingModel | HashingEmbedding":
    """嵌入模型工厂。

    - kind="sentence-transformers"（默认）：语义嵌入，首次使用下载权重（依赖 torch）。
    - kind in {"hash","hashing","local"}：轻量词面嵌入，无需任何重依赖。
    - 若 model_name 以 "ollama://" 开头，则无论 kind 均返回走 Ollama 的 EmbeddingModel。
    """
    if model_name and model_name.startswith("ollama://"):
        return EmbeddingModel(model_name)
    if kind in ("hash", "hashing", "local"):
        return HashingEmbedding()
    return EmbeddingModel(model_name)
