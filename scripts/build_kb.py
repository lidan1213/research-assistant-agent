"""从知识文档目录构建本地 RAG 知识库（分块 -> 向量化 -> FAISS 落盘）并支持检索验证。

用法:
  # 1) 构建（扫描 data/knowledge_docs/ 下所有 .txt/.md，分块后向量化入库）
  python scripts/build_kb.py build

  # 2) 检索验证（对已构建的知识库执行查询，展示 top-k 命中）
  python scripts/build_kb.py search "How does ReAct combine reasoning and acting?"
  python scripts/build_kb.py search "什么是检索增强生成" --top-k 3

依赖:
  - Ollama 本地服务（ollama serve）+ nomic-embed-text 模型
  - 项目 venv 已装 faiss-cpu / numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.knowledge.chunking import Chunker  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.knowledge.embeddings import EmbeddingModel  # noqa: E402
from app.knowledge.retriever import Retriever  # noqa: E402

DOCS_DIR = ROOT / "data" / "knowledge_docs"


def _embedding_spec() -> str:
    """嵌入模型统一读生产配置（默认 BAAI/bge-small-zh-v1.5），避免与 .env 不一致。"""
    return get_settings().knowledge.embedding_model


def load_docs(docs_dir: Path) -> list[dict]:
    """扫描知识文档目录，返回 [{doc_id, text, source}]。"""
    docs: list[dict] = []
    for p in sorted(docs_dir.iterdir()):
        if p.suffix.lower() not in (".txt", ".md"):
            continue
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append({"doc_id": p.stem, "text": text, "source": p.name})
    return docs


def cmd_build(args: argparse.Namespace) -> int:
    docs = load_docs(DOCS_DIR)
    if not docs:
        print(f"[build] 知识文档目录为空: {DOCS_DIR}", file=sys.stderr)
        print("[build] 请先运行 python scripts/fetch_arxiv_docs.py 下载示例文档", file=sys.stderr)
        return 1

    chunker = Chunker(chunk_size=args.chunk_size, overlap=args.overlap)
    embedding = EmbeddingModel(_embedding_spec())
    retriever = Retriever(embedding=embedding, chunker=chunker, use_bm25=True, use_hybrid=True)

    print(f"[build] 扫描到 {len(docs)} 篇文档 -> {DOCS_DIR}")
    print(f"[build] 嵌入后端: {_embedding_spec()} | 分块: chunk_size={args.chunk_size}, overlap={args.overlap}")

    # clear=True 幂等重建，避免重复入库
    count = retriever.ingest_documents(docs, clear=True)
    print(f"[build] 入库完成: {count} 个文本块(chunk) -> {retriever.store.index_path}")
    print(f"[build] 元数据: {retriever.store.meta_path}")

    # 打印分块样例，便于肉眼检查切分质量
    for d in docs:
        chunks = chunker.chunk_doc(d["doc_id"], d["text"], {"source": d["source"]})
        print(f"  {d['doc_id']}: {len(chunks)} 块 | 首块前60字: {chunks[0].text[:60]!r}" if chunks else f"  {d['doc_id']}: 0 块")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    embedding = EmbeddingModel(_embedding_spec())
    retriever = Retriever(embedding=embedding, use_bm25=True, use_hybrid=True)

    if retriever.count() == 0:
        print("[search] 知识库为空，请先运行 python scripts/build_kb.py build", file=sys.stderr)
        return 1

    import asyncio

    hits = asyncio.run(retriever.retrieve(args.query, top_k=args.top_k))
    print(f"\n===== 查询: {args.query} =====")
    print(f"命中 {len(hits)} 条（混合检索: FAISS 向量 + BM25 + RRF 融合 + 重排）\n")
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        src = meta.get("source", "?")
        doc_id = meta.get("doc_id", "?")
        print(f"[{i}] 文档: {doc_id} | 来源: {src} | 相似度: {h.get('score', 0):.4f}")
        print(f"    {h.get('text', '')[:220]}")
        print("-" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 RAG 知识库构建与检索")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="扫描知识文档目录并构建向量知识库")
    p_build.add_argument("--chunk-size", type=int, default=400, help="分块大小（字符数）")
    p_build.add_argument("--overlap", type=int, default=80, help="相邻分块重叠（字符数）")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="检索知识库")
    p_search.add_argument("query", help="查询语句")
    p_search.add_argument("--top-k", type=int, default=3, help="返回条数")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
