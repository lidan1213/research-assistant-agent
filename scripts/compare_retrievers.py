"""RAG 消融实验：多 Embedding × 多 Reranker 在固定 qrels 上的对比评测。

对每种 Embedding 使用独立的临时向量库（避免维度冲突和污染生产 ChromaDB），
在中文集/扩展集上分别评测，输出 Markdown 对比表与 JSON 报告。

用法：
  PYTHONPATH= .venv/Scripts/python.exe scripts/compare_retrievers.py \
    --embeddings "ollama://nomic-embed-text,BAAI/bge-small-zh-v1.5,BAAI/bge-m3" \
    --rerankers "lexical,cross,none" \
    --qrels "data/knowledge/chinese_retrieval_qrels.json,data/knowledge/extended_retrieval_qrels.json"

注意：
  - BAAI 系列 embedding / cross-encoder 需要 torch + sentence-transformers，
    首次运行会下载权重（可设置 HF_ENDPOINT=https://hf-mirror.com 加速国内下载）。
  - bge-m3 显存/内存占用较高，可单独跑。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_docs(docs_dir: Path) -> list[dict]:
    """与 scripts/rebuild_knowledge.py 相同的 loader（txt/md/json/csv）。"""
    import csv

    docs: list[dict] = []
    for path in sorted(docs_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        elif suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            text = raw.get("content") or raw.get("text") or json.dumps(raw, ensure_ascii=False)
        elif suffix == ".csv":
            rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig", errors="replace").splitlines()))
            text = "\n".join("；".join(f"{k}: {v}" for k, v in row.items()) for row in rows)
        else:
            continue
        if text.strip():
            docs.append({"doc_id": path.stem, "source": path.name, "text": text, "metadata": {"source": path.name}})
    return docs


def slug(value: str) -> str:
    return value.replace("://", "_").replace("/", "_").replace(":", "_").replace("-", "_")[:48]


async def run_one(embedding_spec: str, reranker_kind: str, qrels_path: Path,
                  docs: list[dict], ks: list[int], rewriter=None) -> dict:
    from app.eval_retrieval import evaluate_retriever, load_qrels
    from app.knowledge.embeddings import EmbeddingModel
    from app.knowledge.rerank import create_reranker
    from app.knowledge.retriever import Retriever
    from app.knowledge.vectorstore import FAISSVectorStore

    tmp = tempfile.mkdtemp(prefix=f"eval_{slug(embedding_spec)}_")
    store = FAISSVectorStore(index_path=f"{tmp}/idx.faiss")
    retriever = Retriever(
        embedding=EmbeddingModel(embedding_spec),
        store=store,
        top_k=max(ks),
        rerank_k=max(ks) * 2,
        use_hybrid=True,
        reranker=create_reranker(reranker_kind),
        rewriter=rewriter,
    )
    retriever.ingest_documents(docs, clear=True)

    qrels = load_qrels(qrels_path)
    t0 = time.monotonic()
    report = await evaluate_retriever(retriever, qrels, ks)
    elapsed = time.monotonic() - t0
    return {
        "embedding": embedding_spec,
        "reranker": reranker_kind,
        "rewrite": rewriter is not None,
        "dataset": qrels_path.stem,
        "queries": report["query_count"],
        "avg_ms_per_query": round(elapsed / max(len(qrels), 1) * 1000, 1),
        "aggregate": report["aggregate"],
    }


def build_llm_rewriter() -> object:
    """LLM 查询改写器：检索前把 query 改写为更适合召回的形式。

    带按 query 的缓存（消融实验中同一 query 会被多个组合重复检索，避免重复调用 LLM）。
    改写失败静默回退原 query（与生产 Retriever 行为一致）。
    """
    from app.llm.base import ChatMessage, MessageRole
    from app.llm.factory import get_llm

    llm = get_llm()
    cache: dict[str, str] = {}

    async def rewrite(query: str) -> str:
        if query in cache:
            return cache[query]
        try:
            prompt = (
                "你是检索查询改写助手。请把下面的用户问题改写为更适合向量检索的查询："
                "补充专业术语/同义词，去除口语化表达，保持原意，只输出改写后的查询本身。\n"
                f"问题：{query}\n改写："
            )
            resp = await llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
            new = (resp.content or "").strip()
            cache[query] = new if new else query
            return cache[query]
        except Exception:  # noqa: BLE001
            cache[query] = query
            return query

    return rewrite


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 消融对比")
    parser.add_argument("--embeddings", default="ollama://nomic-embed-text,BAAI/bge-small-zh-v1.5")
    parser.add_argument("--rerankers", default="lexical,cross,none")
    parser.add_argument("--qrels", default="data/knowledge/chinese_retrieval_qrels.json,data/knowledge/extended_retrieval_qrels.json")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--docs-dir", default="data/knowledge_docs")
    parser.add_argument("--output", default="data/knowledge/ablation_report.json")
    parser.add_argument("--rewrite", action="store_true",
                        help="开启 LLM 查询改写对照（调用生产 LLM，按 query 缓存）")
    args = parser.parse_args()

    docs = load_docs(ROOT / args.docs_dir)
    qrels_list = [ROOT / p for p in args.qrels.split(",") if p.strip()]
    embeddings = [e.strip() for e in args.embeddings.split(",") if e.strip()]
    rerankers = [r.strip() for r in args.rerankers.split(",") if r.strip()]
    ks = sorted({int(k) for k in args.k})
    rewriter = build_llm_rewriter() if args.rewrite else None

    print(f"docs={len(docs)} embeddings={embeddings} rerankers={rerankers} "
          f"qrels={[q.name for q in qrels_list]} k={ks} rewrite={'ON' if rewriter else 'OFF'}")
    results: list[dict] = []
    for emb in embeddings:
        for rk in rerankers:
            for qp in qrels_list:
                try:
                    r = await run_one(emb, rk, qp, docs, ks, rewriter)
                    results.append(r)
                    agg3 = r["aggregate"].get("3", {})
                    print(f"[OK] {emb.split('/')[-1][:22]:24s} | {rk:7s} | {qp.stem[:10]:10s} | "
                          f"R@3={agg3.get('recall', 0):.3f} MRR@3={agg3.get('mrr', 0):.3f} "
                          f"nDCG@3={agg3.get('ndcg', 0):.3f} | {r['avg_ms_per_query']}ms/q")
                except Exception as e:  # noqa: BLE001
                    results.append({"embedding": emb, "reranker": rk, "dataset": qp.stem,
                                    "rewrite": rewriter is not None, "error": str(e)})
                    print(f"[FAIL] {emb.split('/')[-1][:22]:24s} | {rk:7s} | {qp.stem[:10]:10s} | {e}")

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out_path}")

    # Markdown 对比表（中文集）
    print("\n=== 中文集 @3 对比 ===")
    print("| Embedding | Reranker | Recall@3 | MRR@3 | nDCG@3 | 延迟(ms/q) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if r.get("dataset") != "chinese_retrieval_qrels" or "error" in r:
            continue
        agg = r["aggregate"].get("3", {})
        print(f"| {r['embedding'].split('/')[-1][:28]} | {r['reranker']} | "
              f"{agg.get('recall', 0):.3f} | {agg.get('mrr', 0):.3f} | {agg.get('ndcg', 0):.3f} | {r['avg_ms_per_query']} |")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    asyncio.run(main())
