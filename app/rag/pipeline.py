"""RAG Pipeline：检索管线编排（Query → Rewrite → Retrieve → Fuse → Rerank → Context）。

设计：薄编排层，复用 app/knowledge/ 的底层实现（vectorstore/bm25/rerank/embeddings），
但把「多路召回 + RRF 融合 + 重排 + 上下文构建」串成清晰、可配置、可评测的管线。

与现有 Retriever 的关系：本 Pipeline 是重构后的目标形态；app/knowledge/retriever.py
保留为兼容壳（委托本管线），现有调用方（Agent / API / 评测脚本）不受影响。

参数配置化（可通过 KnowledgeSettings 或构造参数覆盖）：
- top_k: 最终返回条数
- rerank_top_n: 重排候选数（> top_k 才有重排意义）
- rrf_k: RRF 常数
- use_rewrite / use_bm25 / use_rerank: 各环节开关（评测消融用）
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.knowledge.bm25 import BM25Index
from app.knowledge.chunking import Chunk, Chunker
from app.knowledge.embeddings import EmbeddingModel
from app.knowledge.rerank import BaseReranker, create_reranker
from app.knowledge.stores import create_vector_store
from app.rag.fusion import rrf_fuse
from app.rag.types import RetrievalResult

# 查询改写钩子：输入原始 query，返回改写后的 query
Rewriter = Callable[[str], Any]


class RetrievalPipeline:
    """统一检索管线。"""

    def __init__(
        self,
        embedding: EmbeddingModel | None = None,
        store: Any | None = None,
        *,
        top_k: int = 5,
        rerank_top_n: int = 10,
        rrf_k: int = 60,
        use_rewrite: bool = True,
        use_bm25: bool = True,
        use_rerank: bool = True,
        rewriter: Rewriter | None = None,
        reranker: BaseReranker | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        from app.config import get_settings

        s = get_settings().knowledge
        self.embedding = embedding or EmbeddingModel()
        self.store = store or create_vector_store()
        self.top_k = top_k or s.vector_top_k
        self.rerank_top_n = rerank_top_n or s.rerank_top_k
        self.rrf_k = rrf_k
        self.use_rewrite = use_rewrite
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.rewriter = rewriter
        self.reranker = reranker or create_reranker(
            s.reranker_kind,
            model_name=s.reranker_model,
            w_vector=s.rerank_vector_weight,
            w_lexical=s.rerank_lexical_weight,
        )
        self.low_confidence_score = s.low_confidence_score
        self.chunker = chunker or (
            Chunker(s.chunk_size, s.chunk_overlap) if s.chunking_enabled else None
        )

        # BM25 稀疏索引（与向量索引平行）
        self.bm25: BM25Index | None = BM25Index() if use_bm25 else None
        self._doc_by_docid: dict[str, dict] = {}
        if self.bm25 is not None:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        if self.bm25 is None:
            return
        self.bm25.clear()
        self._doc_by_docid = {}
        ids: list[str] = []
        texts: list[str] = []
        for d in self.store._docs:
            did = d.get("metadata", {}).get("doc_id") or str(d.get("id"))
            ids.append(did)
            texts.append(d.get("text", ""))
            self._doc_by_docid[did] = d
        self.bm25.add(ids, texts)

    # ---------- 检索 ----------
    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Query Rewrite → Vector + BM25 → RRF → Rerank → 截断。

        Redis 缓存（L2）：同用户同 query 短 TTL 命中直接返回；知识库变更时由
        user_kb 主动失效（invalidate_user_kb）。Redis 不可用自动降级。
        返回 dict 列表（兼容现有调用方），每项含 retrieval_method 标注。
        """
        import time as _time

        _t0 = _time.monotonic()
        k = top_k or self.top_k
        original_query = query

        # 缓存命中检查（仅对纯检索查询，避免缓存改写后的中间态）
        try:
            from app.tools.knowledge_search import current_kb_name, current_username

            _user = current_username.get() or ""
            _kb_name = current_kb_name.get()
        except Exception:  # noqa: BLE001
            _user = ""
            _kb_name = None
        revision = 0
        if _user:
            try:
                from app.knowledge.versioning import version_store

                revision = version_store.revision(_user, _kb_name)
            except Exception:  # noqa: BLE001
                pass
        # revision 进入 key 后，激活新版本只需递增 revision；旧缓存自然不可见。
        cache_key = f"kb={_kb_name or '__default__'}|rev={revision}|{original_query.strip()}|k={k}"
        try:
            from app.cache.retrieval_cache import cache_get_or_set

            async def _compute() -> list[dict] | None:
                return await self._retrieve_inner(original_query, k, _t0, _user)

            cached = await cache_get_or_set(
                "retrieve", cache_key, _compute, username=_user, ttl=300, empty_ttl=60
            )
            if cached is not None:
                return cached
        except Exception:  # noqa: BLE001
            pass  # 缓存故障不影响检索
        return await self._retrieve_inner(original_query, k, _t0, _user)

    async def _retrieve_inner(
        self, original_query: str, k: int, _t0: float, _user: str = ""
    ) -> list[dict]:
        """实际检索链路（查询改写/多路召回/RRF/重排/日志/排行），与缓存解耦。"""
        query = original_query
        queries = [query]
        if self.use_rewrite and self.rewriter is not None:
            try:
                rewritten = (await self.rewriter(query)).strip()
                if rewritten and rewritten != query:
                    queries.append(rewritten)
            except Exception:  # noqa: BLE001
                pass

        rerank_query = " ".join(dict.fromkeys(" ".join(queries).split()))
        candidate_k = max(k * 3, self.rerank_top_n, k)

        # 1) 向量召回（多路：原查询 + 改写查询）
        vector_hits: list[dict] = []
        for q in queries:
            try:
                q_vec = await asyncio.to_thread(self.embedding.embed_query, q)
                vector_hits.extend(
                    await asyncio.to_thread(self.store.search, q_vec, candidate_k)
                )
            except Exception:  # noqa: BLE001
                continue
        vmap: dict[str, dict] = {}
        for c in vector_hits:
            did = c.get("metadata", {}).get("doc_id") or str(c.get("id"))
            c["doc_id"] = did
            if did not in vmap or (c.get("score") or 0.0) > (vmap[did].get("score") or 0.0):
                vmap[did] = c

        # 2) BM25 稀疏召回 + RRF 融合
        if self.use_bm25 and self.bm25 is not None:
            bm25_hits: list[dict] = []
            for q in queries:
                bm25_hits.extend(self.bm25.search(q, top_k=candidate_k))
            best_bm25: dict[str, dict] = {}
            for hit in bm25_hits:
                if hit["id"] not in best_bm25 or hit["score"] > best_bm25[hit["id"]]["score"]:
                    best_bm25[hit["id"]] = hit
            fused = rrf_fuse(
                [list(vmap.values()), sorted(best_bm25.values(), key=lambda x: x["score"], reverse=True)],
                k=self.rrf_k,
            )
            method = "hybrid_rrf"
        else:
            fused = list(vmap.values())
            method = "vector"

        # 3) 重排 + 截断
        if self.use_rerank:
            results = self.reranker.rerank(rerank_query, fused)[:k]
            method = "reranked" if fused else method
        else:
            results = fused[:k]

        # 4) 结果归一化 + 来源标注
        # score 语义：优先重排后的 score（reranker 输出，与旧 Retriever 一致，
        # 供上层 min_score 阈值过滤）；无重排时回退 rrf_score / 原始向量分。
        out: list[dict] = []
        for r in results:
            meta = r.get("metadata") or {}
            did = r.get("doc_id") or meta.get("doc_id") or r.get("id") or ""
            raw_score = r.get("score")
            if raw_score is None or raw_score == 0.0:
                raw_score = r.get("rrf_score") or 0.0
            out.append(
                {
                    "doc_id": str(did),
                    "chunk_id": str(meta.get("chunk_id") or r.get("chunk_id") or ""),
                    "score": round(float(raw_score), 4),
                    "source": str(meta.get("source") or r.get("source") or did or "?"),
                    "retrieval_method": method,
                    "text": r.get("text", ""),
                    "metadata": {**meta, "doc_id": str(did)},
                }
            )

        # 低置信度标注
        top_score = max((x["score"] for x in out), default=0.0)
        confidence = "low" if top_score < self.low_confidence_score else "normal"
        for x in out:
            x["retrieval_confidence"] = confidence
            x["low_confidence"] = confidence == "low"

        # Ranking measures relevance, not truth. Conservatively mark observable
        # cross-source disagreements so upper layers must disclose them.
        from app.rag.evidence import annotate_evidence

        evidence_report = annotate_evidence(out, original_query)
        for x in out:
            x["evidence_policy"] = evidence_report["policy"]

        # 检索 Trace + 命中来源（管理员据此识别"僵尸文档"与检索质量趋势）
        hit_sources: list[str] = []
        try:
            from app.llm.trace import get_trace_ledger

            for r in out[:5]:
                src = r.get("source") or r.get("doc_id") or "?"
                hit_sources.append(str(src))
            get_trace_ledger().record(
                "rag",
                f"retrieve:{len(out)}",
                success=True,
                duration_ms=int((_time.monotonic() - _t0) * 1000),
                detail=(
                    f"query={original_query[:60]!r} method={method} top={top_score} "
                    f"confidence={confidence} hits={hit_sources}"
                ),
            )
        except Exception:  # noqa: BLE001
            pass

        # Redis 排行榜（热词 + 文档命中），失败静默
        try:
            from app.cache.ranking import rank_incr

            _hot_user = _user or "global"
            if original_query.strip():
                await rank_incr("hot_query", original_query.strip()[:50], 1.0, _hot_user)
            for src in hit_sources:
                await rank_incr("doc_hit", src, 1.0, _hot_user)
        except Exception:  # noqa: BLE001
            pass

        return out

    # ---------- 上下文构建 ----------
    def build_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """把检索结果拼装为 LLM 上下文（带来源标注，可截断）。

        PromptGuard：检索内容属于外部数据，包裹 <retrieved_context> 标记。
        """
        parts: list[str] = []
        used = 0
        for i, c in enumerate(results, 1):
            src = c.get("source") or "知识库"
            text = c.get("text", "")
            if c.get("evidence_conflict"):
                groups = ",".join(c.get("conflict_group_ids") or [])
                block = f"[文档{i} | 来源:{src} | 证据冲突:{groups}]\n{text}"
            else:
                block = f"[文档{i} | 来源:{src}]\n{text}"
            if max_chars and used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    parts.append(block[:remaining] + "\n…(截断)")
                break
            parts.append(block)
            used += len(block)
        body = "\n\n".join(parts)
        return f"<retrieved_context>\n{body}\n</retrieved_context>" if body else ""

    def count(self) -> int:
        return len(self.store._docs)

    def delete_by_metadata(self, field: str, value: str) -> bool:
        """Delete index entries and rebuild BM25 so dense/sparse views agree."""
        deleted = self.store.delete_by_metadata(field, value)
        if deleted and self.bm25 is not None:
            self._rebuild_bm25()
        return bool(deleted)

    def ingest(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Embed and index text fragments without document chunking."""
        items = [
            (text, metadatas[i] if metadatas and i < len(metadatas) else {})
            for i, text in enumerate(texts)
            if text and text.strip()
        ]
        if not items:
            return 0

        clean_texts = [text for text, _ in items]
        clean_metadatas = [metadata for _, metadata in items]
        before = len(self.store._docs)
        vectors = self.embedding.embed(clean_texts)
        documents = [
            {"text": text, "metadata": clean_metadatas[i]}
            for i, text in enumerate(clean_texts)
        ]
        self.store.add(vectors, documents)

        if self.bm25 is not None:
            ids = [
                clean_metadatas[i].get("doc_id") or str(before + i)
                for i in range(len(clean_texts))
            ]
            self.bm25.add(ids, clean_texts)
            for i, document in enumerate(self.store._docs[-len(clean_texts) :]):
                self._doc_by_docid[ids[i]] = document
        return len(documents)

    def ingest_documents(self, docs: list[dict], *, clear: bool = False) -> int:
        """Index structured documents, optionally chunking them first."""
        if clear:
            self.store.clear()
            if self.bm25 is not None:
                self.bm25.clear()
            self._doc_by_docid.clear()

        chunks: list[Chunk] = []
        for index, document in enumerate(docs):
            doc_id = document.get("doc_id") or f"doc_{index}"
            text = document.get("text", "")
            metadata = dict(document.get("metadata") or {})
            if document.get("source"):
                metadata.setdefault("source", document["source"])
            if self.chunker is not None:
                chunks.extend(self.chunker.chunk_doc(doc_id, text, metadata))
            else:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#0",
                        doc_id=doc_id,
                        text=text,
                        metadata=metadata,
                    )
                )

        return self.ingest(
            [chunk.text for chunk in chunks],
            [
                {**chunk.metadata, "doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id}
                for chunk in chunks
            ],
        )
