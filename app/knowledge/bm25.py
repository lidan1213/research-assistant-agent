"""轻量 BM25 稀疏检索（纯 Python，无依赖）。

与向量（稠密）检索互补，用于「混合检索（Hybrid Retrieval）」：
- 稠密向量擅长语义召回，但短 query / 专有名词容易漏召；
- BM25 擅长精确词面/关键词匹配，对术语、缩写、数字稳健；
- 二者通过 RRF（Reciprocal Rank Fusion）融合，鲁棒性显著优于单一通道。

中文按「字符级 + 英文/数字词级」切分，与项目其它模块的 token 方案保持一致。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

_TOKEN_RE = re.compile(r"[一-鿿]|[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.docs_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}

    def add(self, ids: List[str], texts: List[str]) -> None:
        """增量加入一批文档（与向量库并行维护）。"""
        for doc_id, text in zip(ids, texts):
            toks = tokenize(text)
            self.doc_ids.append(doc_id)
            self.docs_tokens.append(toks)
            self.doc_lens.append(len(toks))
        self._compute_idf()

    def _compute_idf(self) -> None:
        n = len(self.doc_ids)
        if n == 0:
            self.avgdl = 0.0
            self.idf = {}
            return
        df: Dict[str, int] = {}
        for toks in self.docs_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self.idf = {
            w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }
        self.avgdl = sum(self.doc_lens) / n

    def search(self, query: str, top_k: int = 8) -> List[dict]:
        """返回 [{"id","score"}]，按 BM25 得分降序。"""
        if not self.doc_ids:
            return []
        q_toks = tokenize(query)
        if not q_toks:
            return []
        scores: Dict[str, float] = {}
        for q in set(q_toks):
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, toks in enumerate(self.docs_tokens):
                f = toks.count(q)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1))
                scores[self.doc_ids[i]] = scores.get(self.doc_ids[i], 0.0) + idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id": i, "score": float(s)} for i, s in ranked[:top_k]]

    def clear(self) -> None:
        self.doc_ids.clear()
        self.docs_tokens.clear()
        self.doc_lens.clear()
        self.avgdl = 0.0
        self.idf.clear()
