"""文档分块（Chunking）模块。

把长文档切分为语义/结构完整的片段，是高质量 RAG 的前提：
- 递归切分（recursive split）：按段落/句子/标点逐级切，尽量不切断语义单元；
- 滑动窗口合并：相邻片段按 chunk_size 合并，并保留 overlap 重叠区减少边界信息丢失；
- 每个 Chunk 携带稳定 chunk_id 与父文档 doc_id，便于评测 qrels 对齐。

纯 Python，无外部依赖；chunk_size / overlap 以「字符数」为近似单位。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# 默认分隔符优先级：段落 > 换行 > 句末标点 > 逗号 > 空格
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", ";", ".", "！", "？", "，", ",", " "]


class Chunker:
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 80,
        separators: List[str] | None = None,
    ) -> None:
        self.chunk_size = max(50, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size // 2))
        self.separators = separators or DEFAULT_SEPARATORS

    # ---------- 递归切分为「原子片段」 ----------
    def _split_recursive(self, text: str, depth: int) -> List[str]:
        if depth >= len(self.separators) or len(text) <= self.chunk_size:
            return [text] if text else []
        sep = self.separators[depth]
        parts = text.split(sep)
        pieces: List[str] = []
        for i, part in enumerate(parts):
            piece = part + sep if i < len(parts) - 1 and sep else part
            if not piece:
                continue
            if len(piece) > self.chunk_size:
                # 单段仍过长 -> 进入更细粒度的分隔符继续递归
                pieces.extend(self._split_recursive(piece, depth + 1))
            else:
                pieces.append(piece)
        return pieces

    # ---------- 合并为带重叠的窗口 ----------
    def _pack(self, atomic: List[str]) -> List[str]:
        chunks: List[str] = []
        buf = ""
        for piece in atomic:
            if not buf:
                buf = piece
            elif len(buf) + len(piece) <= self.chunk_size:
                buf += piece
            else:
                chunks.append(buf)
                tail = buf[-self.overlap :] if self.overlap else ""
                buf = tail + piece
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        atomic = self._split_recursive(text, 0)
        return self._pack(atomic)

    # ---------- 文档级入口 ----------
    def chunk_doc(
        self, doc_id: str, text: str, metadata: dict | None = None
    ) -> List[Chunk]:
        meta = dict(metadata or {})
        pieces = self.split_text(text)
        if not pieces:
            return []
        return [
            Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, text=p, metadata=meta)
            for i, p in enumerate(pieces)
        ]

    def chunk_documents(self, corpus: List[dict]) -> List[Chunk]:
        """corpus 每项: {"doc_id","text","metadata"?}。返回扁平化 Chunk 列表。"""
        out: List[Chunk] = []
        for idx, d in enumerate(corpus):
            doc_id = d.get("doc_id") or f"doc_{idx}"
            out.extend(self.chunk_doc(doc_id, d.get("text", ""), d.get("metadata")))
        return out


def _coarse_words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()))
