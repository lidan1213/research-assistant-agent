"""RAG 检索器：串起「分块 -> 稠密向量召回 + BM25 稀疏召回 -> 混合融合 -> 重排 -> 上下文拼装」。

设计要点：
1. 支持 doc_id 级别的稳定入库（便于评测 qrels 对齐）与可选「文档分块（chunking）」。
2. 默认启用**混合检索（Hybrid Retrieval）**：向量召回与 BM25 召回通过 RRF 融合，
   兼顾语义召回与精确词面召回，鲁棒性优于单一通道。
3. 重排器可插拔：默认 `LexicalReranker`（当前语料质量/延迟更优），可切换为 CrossEncoder 精排。
4. 可选「查询改写（query rewrite）」：在检索前用 LLM 对 query 做扩展/澄清（异步可注入）。
5. BM25 索引与向量库并行维护；从磁盘加载向量库后自动重建 BM25，保证 build/eval 一致性。
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List

from app.config import get_settings
from app.knowledge.bm25 import BM25Index
from app.knowledge.chunking import Chunker
from app.knowledge.embeddings import EmbeddingModel
from app.knowledge.rerank import BaseReranker, LexicalReranker, create_reranker
from app.knowledge.stores import create_vector_store
from app.knowledge.stores.chroma_local import ChromaVectorStore
from app.knowledge.stores.faiss import FAISSVectorStore

# 查询改写钩子：输入原始 query，返回改写后的 query
Rewriter = Callable[[str], Awaitable[str]]


class Retriever:
    def __init__(
        self,
        embedding: EmbeddingModel | None = None,
        store: "FAISSVectorStore | ChromaVectorStore | None" = None,
        *,
        top_k: int | None = None,
        rerank_k: int | None = None,
        chunker: Chunker | None = None,
        use_bm25: bool = True,
        use_hybrid: bool = True,
        reranker: BaseReranker | None = None,
        rewriter: Rewriter | None = None,
    ) -> None:
        s = get_settings().knowledge
        self.embedding = embedding or EmbeddingModel()
        self.store = store or create_vector_store()
        self.top_k = top_k or s.vector_top_k
        self.rerank_k = rerank_k or s.rerank_top_k
        self.chunker = chunker or (
            Chunker(s.chunk_size, s.chunk_overlap) if s.chunking_enabled else None
        )
        self.use_bm25 = use_bm25
        self.use_hybrid = use_hybrid
        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = create_reranker(
                s.reranker_kind,
                model_name=s.reranker_model,
                w_vector=s.rerank_vector_weight,
                w_lexical=s.rerank_lexical_weight,
            )
        self.rewriter = rewriter
        s = get_settings().knowledge
        self.low_confidence_score = s.low_confidence_score
        self.min_score = s.retrieval_min_score

        # BM25 稀疏索引（与向量索引平行）
        self.bm25: BM25Index | None = BM25Index() if use_bm25 else None
        # doc_id -> 完整 doc 的映射，用于把 BM25-only 命中补齐为候选文档
        self._doc_by_docid: dict[str, dict] = {}
        if self.bm25 is not None:
            self._rebuild_bm25()

        # 统一检索管线（RAG Pipeline）：本类为兼容壳，retrieve 委托 pipeline。
        # 共享同一 store / bm25 / reranker / rewriter，保证入库与检索一致性。
        from app.rag.pipeline import RetrievalPipeline

        self._pipeline = RetrievalPipeline(
            embedding=self.embedding,
            store=self.store,
            top_k=self.top_k,
            rerank_top_n=self.rerank_k,
            use_rewrite=True,
            use_bm25=self.use_hybrid,
            use_rerank=True,
            rewriter=self.rewriter,
            reranker=self.reranker,
            chunker=self.chunker,
        )
        # 共享 BM25 索引与 doc 映射（pipeline 不再重复构建）
        self._pipeline.bm25 = self.bm25
        self._pipeline._doc_by_docid = self._doc_by_docid

    # ---------- BM25 维护 ----------
    def _rebuild_bm25(self) -> None:
        """从已加载的向量库重建 BM25（保证 build 之后独立 eval 的一致性）。"""
        if self.bm25 is None:
            return
        self._doc_by_docid = {}
        ids: List[str] = []
        texts: List[str] = []
        for d in self.store._docs:
            did = d.get("metadata", {}).get("doc_id") or str(d.get("id"))
            ids.append(did)
            texts.append(d.get("text", ""))
            self._doc_by_docid[did] = d
        self.bm25.add(ids, texts)

    # ---------- 入库 ----------
    def ingest(self, texts: List[str], metadatas: List[dict] | None = None) -> int:
        """向量化并写入知识库，返回新增文档数（不分块，每个 text 视为一篇）。"""
        return self._pipeline.ingest(texts, metadatas)

    def ingest_documents(
        self, docs: List[dict], *, clear: bool = False
    ) -> int:
        """结构化入库：docs 每项 {"doc_id","text","source"?,"metadata"?}。

        若设置了 chunker，则先分块（chunk_id 形如 `doc_id#0`），再批量向量化与 BM25 写入。
        返回新增的 chunk/doc 数量。
        """
        return self._pipeline.ingest_documents(docs, clear=clear)

    # ---------- 检索 ----------
    async def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """查询改写 -> 向量召回 -> (BM25 召回) -> RRF 融合 -> 重排 -> 截断。

        委托统一检索管线（app.rag.pipeline.RetrievalPipeline）：
        Redis 缓存（L2）命中直接返回；知识库变更由 user_kb 主动失效；
        Redis 不可用自动降级；结果含 retrieval_method 标注。
        """
        return await self._pipeline.retrieve(query, top_k)

    async def _retrieve_inner(self, query: str, k: int, _t0: float) -> List[dict]:
        """实际检索链路（查询改写/多路召回/RRF/重排/日志），与缓存解耦。"""
        try:
            from app.tools.knowledge_search import current_username

            _user = current_username.get() or ""
        except Exception:  # noqa: BLE001
            _user = ""
        queries = [query]
        if self.rewriter is not None:
            try:
                rewritten = (await self.rewriter(query)).strip()
                if rewritten and rewritten != query:
                    queries.append(rewritten)
                    query = rewritten
            except Exception:  # noqa: BLE001
                pass  # 改写失败不影响主流程

        # 最终重排使用原始查询和改写查询的主题词并集，避免改写结果过度偏移。
        rerank_query = " ".join(dict.fromkeys(" ".join(queries).split()))

        # 多路召回：原始 query 保留用户词面，改写 query 补充专业术语；
        # 任一路失败都回退到另一条，不让查询改写成为单点故障。
        candidate_k = max(k * 3, self.rerank_k, k)
        vector_hits = []
        for q in queries:
            try:
                q_vec = await asyncio.to_thread(self.embedding.embed_query, q)
                vector_hits.extend(await asyncio.to_thread(self.store.search, q_vec, candidate_k))
            except Exception:  # noqa: BLE001
                continue
        vmap: dict[str, dict] = {}
        for c in vector_hits:
            did = c.get("metadata", {}).get("doc_id") or str(c.get("id"))
            c["doc_id"] = did
            # 同一文档可能由原始/改写查询重复召回，保留最高向量分。
            if did not in vmap or (c.get("score") or 0.0) > (vmap[did].get("score") or 0.0):
                vmap[did] = c

        if self.use_hybrid and self.bm25 is not None:
            bm25_hits = []
            for q in queries:
                bm25_hits.extend(self.bm25.search(q, top_k=candidate_k))
            # 同一文档在多路 BM25 中保留最高分，避免重复候选污染 RRF。
            best_bm25: dict[str, dict] = {}
            for hit in bm25_hits:
                if hit["id"] not in best_bm25 or hit["score"] > best_bm25[hit["id"]]["score"]:
                    best_bm25[hit["id"]] = hit
            fused = self._rrf(vmap, sorted(best_bm25.values(), key=lambda x: x["score"], reverse=True))
        else:
            fused = list(vector_hits)

        results = self.reranker.rerank(rerank_query, fused)[:k]
        # 低置信度元数据：上层 Agent 可据此提示“知识库证据有限”，但不强制丢弃结果。
        top_score = max((r.get("score") or 0.0 for r in results), default=0.0)
        confidence = "low" if top_score < self.low_confidence_score else "normal"
        for r in results:
            r["retrieval_confidence"] = confidence
            r["low_confidence"] = confidence == "low"

        # 检索可观测性（RAG 命中日志）：query -> 命中文档/最高相似度/耗时，写 Trace。
        # 管理员据此识别"僵尸文档"（永远检索不到）与检索质量趋势。
        try:
            from app.llm.trace import get_trace_ledger

            top_score = round(max((r.get("score") or 0) for r in results), 3) if results else 0.0
            hit_sources = []
            for r in results[:5]:
                src = (r.get("metadata") or {}).get("source") or (r.get("metadata") or {}).get("doc_id") or "?"
                hit_sources.append(str(src))
            get_trace_ledger().record(
                "rag",
                f"retrieve:{len(results)}",
                success=True,
                duration_ms=int((_time.monotonic() - _t0) * 1000),
                detail=f"query={original_query[:60]!r} rewritten={query[:60]!r} top={top_score} confidence={confidence} hits={hit_sources}",
            )
        except Exception:  # noqa: BLE001
            pass  # 日志失败不影响检索

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

        return results

    def _rrf(self, vmap: dict[str, dict], bm25_hits: List[dict], k: int = 60) -> List[dict]:
        """Reciprocal Rank Fusion：把向量与 BM25 的排名融合为单一候选列表。"""
        scores: dict[str, float] = {}
        payload: dict[str, dict] = {}
        for rank, c in enumerate(vmap.values()):
            did = c["doc_id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
            payload[did] = c
        for rank, h in enumerate(bm25_hits):
            did = h["id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
            if did not in payload:
                # BM25-only 命中：从已索引 doc 补齐完整候选
                payload[did] = self._doc_by_docid.get(
                    did, {"doc_id": did, "text": "", "metadata": {}, "score": 0.0}
                )
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [payload[did] for did, _ in ordered]

    # ---------- 统计与上下文 ----------
    def count(self) -> int:
        """当前知识库中的文档（chunk）数量。"""
        return self._pipeline.count()

    def delete_by_metadata(self, field: str, value: str) -> bool:
        """删除向量与 BM25 中匹配的派生索引条目。"""
        return self._pipeline.delete_by_metadata(field, value)

    def format_context(self, candidates: List[dict]) -> str:
        return self._pipeline.build_context(candidates)
