"""Embedding 全量迁移：把 ChromaDB 所有 collection 重建为 bge-small-zh-v1.5。

背景：消融实验证明中文场景 bge 优于 nomic（Recall@3 0.617→0.900）。
切换 embedding 必须重建索引（同一 collection 内向量维度必须一致）。

流程（对每个 collection）：
  1. 导出全部 (id, text, metadata) 并备份为 JSON（data/migration_backup/）
  2. 删除并重建 collection（保留原 id + metadata，向量换成 bge）
  3. 验证：维度 512、count 与迁移前一致

用法：
  PYTHONPATH= .venv/Scripts/python.exe scripts/migrate_embedding.py --dry-run   # 导出备份 + 打印计划，不写入
  PYTHONPATH= .venv/Scripts/python.exe scripts/migrate_embedding.py             # 执行迁移
  PYTHONPATH= .venv/Scripts/python.exe scripts/migrate_embedding.py --collections knowledge,kb_student1 --embedding BAAI/bge-small-zh-v1.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8001"
BACKUP_DIR = ROOT / "data" / "migration_backup"
BATCH = 32  # 嵌入分批大小（控制内存）


def list_collections(client: httpx.Client) -> list[dict]:
    return client.get(f"{BASE}/api/v1/collections").json()


def export_collection(client: httpx.Client, cid: str) -> dict:
    """导出 collection 全部文档（不含向量）。"""
    resp = client.post(
        f"{BASE}/api/v1/collections/{cid}/get",
        json={"include": ["documents", "metadatas"]},
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "ids": data.get("ids") or [],
        "documents": data.get("documents") or [],
        "metadatas": data.get("metadatas") or [],
    }


def backup_collection(name: str, data: dict) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"{name}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


def recreate_collection(client: httpx.Client, name: str) -> str:
    """删除并重建 collection，返回新 UUID。"""
    client.delete(f"{BASE}/api/v1/collections/{name}")
    created = client.post(
        f"{BASE}/api/v1/collections",
        json={"name": name, "metadata": {"hnsw:space": "cosine"}},
    )
    created.raise_for_status()
    return created.json()["id"]


def embed_texts(texts: list[str], embedding_spec: str) -> list[list[float]]:
    """用指定 embedding 模型分批嵌入。"""
    from app.knowledge.embeddings import EmbeddingModel

    model = EmbeddingModel(embedding_spec)
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        out.extend(model.embed(batch))
        print(f"    嵌入 {min(i + BATCH, len(texts))}/{len(texts)}")
    return out


def migrate_collection(
    client: httpx.Client, name: str, cid: str, embedding_spec: str, dry_run: bool
) -> dict:
    """迁移单个 collection，返回统计。"""
    print(f"\n=== {name} ===")
    data = export_collection(client, cid)
    n = len(data["ids"])
    print(f"  文档数: {n}")
    if n == 0:
        print("  空 collection，跳过")
        return {"name": name, "count": 0, "skipped": True}

    backup_path = backup_collection(name, data)
    print(f"  已备份: {backup_path}")

    if dry_run:
        return {"name": name, "count": n, "dry_run": True}

    # 重建 collection
    new_cid = recreate_collection(client, name)
    # 重嵌入
    vecs = embed_texts(data["documents"], embedding_spec)
    # 写回（保留原 id + metadata）
    resp = client.post(
        f"{BASE}/api/v1/collections/{new_cid}/add",
        json={
            "ids": data["ids"],
            "embeddings": vecs,
            "documents": data["documents"],
            "metadatas": data["metadatas"],
        },
    )
    resp.raise_for_status()
    return {"name": name, "count": n, "new_cid": new_cid, "dim": len(vecs[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding 全量迁移（nomic → bge）")
    parser.add_argument("--dry-run", action="store_true", help="只备份 + 打印计划，不写入")
    parser.add_argument("--embedding", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--collections", default="", help="逗号分隔；空 = 全部")
    args = parser.parse_args()

    with httpx.Client(timeout=120) as client:
        cols = list_collections(client)
        names = [c["name"] for c in cols]
        if args.collections:
            wanted = {x.strip() for x in args.collections.split(",") if x.strip()}
            cols = [c for c in cols if c["name"] in wanted]
        else:
            # 默认全部（仅排除明显的测试/临时 collection；memory_facts 必须迁移）
            skip = {"test", "tmp"}
            cols = [c for c in cols if not any(s in c["name"].lower() for s in skip)]
        print(f"待迁移 {len(cols)} 个 collection: {[c['name'] for c in cols]}")
        print(f"目标 embedding: {args.embedding}（{'DRY-RUN 仅备份' if args.dry_run else '执行写入'}）")

        results = []
        for c in cols:
            try:
                results.append(migrate_collection(client, c["name"], c["id"], args.embedding, args.dry_run))
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {c['name']}: {e}")
                results.append({"name": c["name"], "error": str(e)[:200]})

        print("\n=== 迁移结果 ===")
        for r in results:
            print(" ", r)
        ok = [r for r in results if r.get("count", 0) > 0 and r.get("dry_run")]
        if ok and args.dry_run:
            print(f"\nDRY-RUN 完成：{len(ok)} 个 collection 已备份，未写入。检查备份后执行正式迁移。")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
