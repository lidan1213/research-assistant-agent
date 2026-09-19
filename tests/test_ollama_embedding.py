"""Ollama 嵌入接入测试：验证 `ollama://` 前缀分发、host 解析、请求体格式与批量/回退逻辑。

无需真实 Ollama 服务，用 unittest.mock 替换 urllib.request.urlopen。
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import urllib.error

from app.knowledge.embeddings import (
    EmbeddingModel,
    OllamaEmbedding,
    create_embedding,
)


def _fake_resp(obj):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(obj).encode()
    return cm


def test_ollama_spec_default_host():
    e = OllamaEmbedding("ollama://nomic-embed-text")
    assert e.model == "nomic-embed-text"
    assert e.host == "http://localhost:11434"


def test_ollama_spec_with_host():
    e = OllamaEmbedding("ollama://127.0.0.1:9999/foo")
    assert e.model == "foo"
    assert e.host == "http://127.0.0.1:9999"


def test_dispatch_ollama():
    m = EmbeddingModel("ollama://nomic-embed-text")
    assert m._ollama is not None and m._ollama.model == "nomic-embed-text"


def test_dispatch_local_default():
    m = EmbeddingModel("BAAI/bge-small-zh-v1.5")
    assert m._ollama is None


def test_create_embedding_ollama_prefix():
    m = create_embedding(model_name="ollama://nomic-embed-text")
    assert isinstance(m, EmbeddingModel) and m._ollama is not None


def test_ollama_embed_query_payload():
    with patch(
        "urllib.request.urlopen", return_value=_fake_resp({"embedding": [0.1, 0.2]})
    ) as p:
        e = OllamaEmbedding("ollama://nomic-embed-text")
        out = e.embed_query("hi")
    assert out == [0.1, 0.2]
    req = p.call_args.args[0]
    body = json.loads(req.data.decode())
    assert body == {"model": "nomic-embed-text", "prompt": "hi"}


def test_ollama_embed_batch():
    with patch(
        "urllib.request.urlopen",
        return_value=_fake_resp({"embeddings": [[0.1], [0.2]]}),
    ) as p:
        e = OllamaEmbedding("ollama://nomic-embed-text")
        out = e.embed(["a", "b"])
    assert out == [[0.1], [0.2]]
    body = json.loads(p.call_args.args[0].data.decode())
    assert body == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_ollama_embed_batch_fallback():
    calls = []

    def side(req, timeout=None):
        body = json.loads(req.data.decode())
        calls.append(body)
        if "input" in body:
            raise urllib.error.URLError("no batch")
        return _fake_resp({"embedding": [0.9]})

    with patch("urllib.request.urlopen", side_effect=side):
        e = OllamaEmbedding("ollama://nomic-embed-text")
        out = e.embed(["a", "b"])
    assert out == [[0.9], [0.9]]
    assert len(calls) == 3  # 1 失败批量 + 2 逐条
    assert "input" in calls[0]
    assert all("prompt" in c for c in calls[1:])
