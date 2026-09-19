"""从 data/knowledge_docs 重建全局 knowledge collection，支持 txt/md/json/csv。"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from app.api.deps import get_retriever
from app.knowledge.chunking import Chunker

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "knowledge_docs"


def load_documents() -> list[dict]:
    docs: list[dict] = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                text = raw.get("content") or raw.get("text") or json.dumps(raw, ensure_ascii=False)
            else:
                text = json.dumps(raw, ensure_ascii=False)
        elif path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig", errors="replace").splitlines()))
            text = "\n".join("；".join(f"{k}: {v}" for k, v in row.items()) for row in rows)
        else:
            continue
        if text.strip():
            docs.append({
                "doc_id": path.stem,
                "source": path.name,
                "text": text,
                "metadata": {"source": path.name, "format": path.suffix.lower().lstrip(".")},
            })
    return docs


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=0, help="启用结构化分块；0 表示整篇文档入库")
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()
    docs = load_documents()
    retriever = get_retriever()
    if args.chunk_size > 0:
        retriever.chunker = Chunker(args.chunk_size, args.overlap)
    count = retriever.ingest_documents(docs, clear=True)
    print(f"loaded_files={len(docs)} ingested_chunks={count} total={retriever.count()}")
    for doc in docs:
        print(f"- {doc['doc_id']} ({doc['metadata']['format']})")


if __name__ == "__main__":
    asyncio.run(main())
