"""可插拔重排器（Reranker）。

检索召回的候选经过重排后再截断，是提升 RAG 上下文质量的关键一步：
- `LexicalReranker`（默认）：融合「向量相似度 + 词面重叠」，CPU 可跑、零额外依赖；
- `CrossEncoderReranker`：基于 sentence-transformers 的 CrossEncoder 精排，
  首次使用懒加载；加载失败时降级为 `LexicalReranker`。

`create_reranker(kind)` 工厂便于在配置或 API 中切换。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List

from app.core.logging import get_logger


logger = get_logger("knowledge.rerank")

_TOK_RE = re.compile(r"[\w\u4e00-\u9fff]+")
# 英文停用词不携带主题信息，若参与词面重排会让包含 "and/is/what" 的无关论文
# 获得虚高分（尤其是英文科研文档）。中文不做停用词删除，避免误伤专有词。
_EN_STOPWORDS = {
    "a", "an", "the", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "their",
    "this", "to", "use", "what", "when", "where", "which", "who", "with", "through",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOK_RE.findall((text or "").lower())
        if token not in _EN_STOPWORDS
    }


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """对候选（含 text / score / metadata 等）重排序，返回同结构列表。"""
        raise NotImplementedError

    def __call__(self, query: str, candidates: List[dict]) -> List[dict]:
        return self.rerank(query, candidates)


class LexicalReranker(BaseReranker):
    """默认重排器：向量余弦分 + 词面重叠的加权融合（均归一化到 [0,1]）。"""

    def __init__(self, w_vector: float = 0.6, w_lexical: float = 0.4) -> None:
        self.w_vector = w_vector
        self.w_lexical = w_lexical

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return candidates
        q = _tokens(query)
        max_vec = max((c.get("score", 0.0) for c in candidates), default=1.0) or 1.0

        def score(c: dict) -> float:
            vs = c.get("score", 0.0) / max_vec
            overlap = len(q & _tokens(c.get("text", "")))
            return self.w_vector * vs + self.w_lexical * overlap

        return sorted(candidates, key=score, reverse=True)


class CrossEncoderReranker(BaseReranker):
    """CrossEncoder 语义精排；模型不可用时自动降级为轻量重排。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        fallback: BaseReranker | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = fallback or LexicalReranker()
        self._disabled = False

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return candidates
        if self._disabled:
            return self._fallback.rerank(query, candidates)
        try:
            model = self._ensure()
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            return sorted(candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        except Exception as exc:  # noqa: BLE001
            # 下载失败、缺少 torch 或模型不兼容都不应中断 RAG 主链路。
            self._disabled = True
            logger.warning("Cross-Encoder 加载/推理失败，降级为 lexical reranker: %s", exc)
            return self._fallback.rerank(query, candidates)


class IdentityReranker(BaseReranker):
    """透传重排器：保持输入顺序不变。

    用于「禁用重排」的消融对比——直接采用 RRF 融合后的顺序，便于与
    启用 LexicalReranker / CrossEncoderReranker 的结果做横向对照。
    """

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        return list(candidates)


def create_reranker(kind: str = "lexical", **kwargs) -> BaseReranker:
    """重排器工厂。

    - kind in {"lexical","default"}：词面融合重排（默认）。
    - kind in {"cross","cross-encoder","crossencoder"}：CrossEncoder 精排。
    - kind in {"none","identity"}：透传（禁用重排）。
    """
    model_name = kwargs.pop("model_name", "BAAI/bge-reranker-v2-m3")
    if kind in ("cross", "cross-encoder", "crossencoder"):
        # 各重排器参数不同：不要把 lexical 权重透传给 CrossEncoder 构造器。
        fallback = LexicalReranker(
            w_vector=kwargs.pop("w_vector", 0.6),
            w_lexical=kwargs.pop("w_lexical", 0.4),
        )
        return CrossEncoderReranker(model_name=model_name, fallback=fallback)
    if kind in ("none", "identity"):
        return IdentityReranker()
    return LexicalReranker(**kwargs)
