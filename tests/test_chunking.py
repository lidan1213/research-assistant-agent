"""文档分块（Chunking）单元测试。"""
from __future__ import annotations

from app.knowledge.chunking import Chunker


def test_chunker_recursive_split_and_overlap():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = "。".join(f"第{i}句话的内容描述" for i in range(20))
    chunks = chunker.chunk_doc("doc1", text)

    assert len(chunks) > 1
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.chunk_id.startswith("doc1#") for c in chunks)
    # 分块后拼接长度应接近原文（overlap 仅少量重叠）
    joined = "".join(c.text for c in chunks)
    assert len(joined) >= len(text)


def test_chunker_short_text_single_chunk():
    chunker = Chunker()
    chunks = chunker.chunk_doc("d", "一句很短的话。")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d#0"


def test_chunk_documents_flatten():
    corpus = [
        {"doc_id": "a", "text": "机器学习 深度学习。" * 20},
        {"doc_id": "b", "text": "足球 世界杯。" * 20},
    ]
    chunks = Chunker(chunk_size=100, overlap=20).chunk_documents(corpus)
    assert len(chunks) >= 2
    assert {c.doc_id for c in chunks} == {"a", "b"}
