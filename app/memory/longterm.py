"""长期记忆（跨会话持久化）：L2 事实主存储为 ChromaDB，SQLite 作为降级路径。

存储结构：
- L3 会话索引（SQLite sessions 表）：id/title/summary/created_at/updated_at —— 结构化元数据，留在 SQLite。
- L2 跨会话事实（ChromaDB collection `memory_facts`，主存储）：
    document = 事实原文；metadata = {session_id, category, tags(逗号拼接), created_at}；
    vector   = EmbeddingModel（默认 ollama://nomic-embed-text，768 维）。
- 降级路径（SQLite facts 表 + FTS5/LIKE）：ChromaDB 服务不可用时自动回退，
  保证离线环境 / 测试 / CI 不依赖 ChromaDB 也能运行（与项目「失败降级」哲学一致）。

设计目标：让多 Agent 一次研究的「黑板结论 / 链路成品」在运行结束后仍可被后续会话
语义召回，实现「记忆的链路」从「运行时工作记忆」到「长期记忆」的沉淀与跨会话复用。
"""
from __future__ import annotations

import os
import hashlib
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("memory.longterm")

CHROMA_COLLECTION = "memory_facts"
LEGACY_USER = "legacy"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def username_from_session(session_id: str) -> str:
    """Derive the authenticated owner from the canonical username__session id."""
    value = str(session_id or "")
    return value.split("__", 1)[0] if "__" in value and value.split("__", 1)[0] else LEGACY_USER


def _normalize_fact(content: str) -> str:
    return " ".join((content or "").strip().lower().split())


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalize_fact(content).encode("utf-8")).hexdigest()


class LongTermMemory:
    """长期记忆库：L3 会话索引（SQLite）+ L2 事实（ChromaDB 主存储，SQLite 降级）。"""

    def __init__(self, path: Optional[str] = None, *, chroma: bool = True) -> None:
        self.path = path or get_settings().memory.longterm_path
        self.chroma_enabled = chroma
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        # ChromaDB 连接与嵌入模型（失败则置 None，所有 L2 读写自动降级 SQLite）
        self._chroma: dict | None = None
        self._embedding = None
        if self.chroma_enabled:
            self._init_chroma()

    # ---------- 存储初始化 ----------
    def _init_schema(self) -> None:
        cur = self._conn
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sessions("
            "id TEXT PRIMARY KEY, title TEXT, summary TEXT, "
            "created_at TEXT, updated_at TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS facts("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, category TEXT, "
            "content TEXT, tags TEXT, created_at TEXT)"
        )
        additions = {
            "username": "TEXT NOT NULL DEFAULT 'legacy'",
            "content_hash": "TEXT NOT NULL DEFAULT ''",
            "memory_key": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "confidence": "REAL NOT NULL DEFAULT 1.0",
            "expires_at": "TEXT NOT NULL DEFAULT ''",
            "supersedes_id": "INTEGER",
        }
        existing = {row[1] for row in cur.execute("PRAGMA table_info(facts)").fetchall()}
        for name, ddl in additions.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE facts ADD COLUMN {name} {ddl}")
        # Backfill ownership and hashes for pre-upgrade rows. Unprefixed sessions
        # remain `legacy` and are deliberately never recalled for signed-in users.
        rows = cur.execute("SELECT id, session_id, content, username, content_hash FROM facts").fetchall()
        for rowid, sid, content, owner, digest in rows:
            inferred = username_from_session(sid)
            cur.execute(
                "UPDATE facts SET username=?, content_hash=? WHERE id=?",
                (inferred if owner in (None, "", LEGACY_USER) else owner, digest or _content_hash(content), rowid),
            )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_owner_status ON facts(username, status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_owner_hash ON facts(username, content_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_owner_key ON facts(username, memory_key)")
        # 优先尝试 FTS5 全文索引；失败则回退 LIKE
        self._fts = False
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts "
                "USING fts5(content)"
            )
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False
            logger.warning("当前 SQLite 不支持 FTS5，长期记忆检索回退为 LIKE 关键字匹配")
        cur.commit()

    def _init_chroma(self) -> None:
        """连接 ChromaDB 服务端（L2 主存储）。失败则 self._chroma=None，降级 SQLite。"""
        try:
            import httpx

            from app.knowledge.embeddings import EmbeddingModel

            base = os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
            http = httpx.Client(timeout=httpx.Timeout(3.0, connect=1.0))
            cols = http.get(f"{base}/api/v1/collections").json()
            cid = next((c["id"] for c in cols if c.get("name") == CHROMA_COLLECTION), None)
            if not cid:
                created = http.post(
                    f"{base}/api/v1/collections",
                    json={"name": CHROMA_COLLECTION, "metadata": {"hnsw:space": "cosine"}},
                ).json()
                cid = created["id"]
            self._chroma = {"base": base, "cid": cid, "http": http}
            self._embedding = EmbeddingModel()
            self._migrate_legacy_facts()
            self._migrate_chroma_metadata()
            logger.info(f"长期记忆 L2 使用 ChromaDB: {base} collection={CHROMA_COLLECTION}")
        except Exception as e:  # noqa: BLE001
            self._chroma = None
            self._embedding = None
            logger.warning(f"ChromaDB 不可用，长期记忆 L2 降级 SQLite: {e}")

    def _migrate_legacy_facts(self) -> None:
        """一次性迁移：ChromaDB collection 为空但 SQLite facts 表有历史数据时灌入。"""
        if self._chroma is None:
            return
        try:
            base, cid, http = self._chroma["base"], self._chroma["cid"], self._chroma["http"]
            got = http.post(
                f"{base}/api/v1/collections/{cid}/get",
                json={"include": ["metadatas"]},
            ).json()
            existing = got.get("ids") or []
            if existing:
                return  # ChromaDB 已有数据，不重复迁移
            rows = self._conn.execute(
                "SELECT id, session_id, category, content, tags, created_at, username, status, confidence, expires_at, memory_key "
                "FROM facts ORDER BY id"
            ).fetchall()
            if not rows:
                return
            ids, docs, metas = [], [], []
            for r in rows:
                rowid, sid, cat, content, tags, created, owner, status, confidence, expires_at, memory_key = r
                ids.append(f"fact_{rowid}")
                docs.append(content)
                metas.append(
                    {
                        "session_id": sid,
                        "category": cat,
                        "tags": tags or "",
                        "created_at": created,
                        "username": owner or username_from_session(sid),
                        "status": status or "active",
                        "confidence": float(confidence or 0),
                        "expires_at": expires_at or "",
                        "memory_key": memory_key or "",
                    }
                )
            http.post(
                f"{base}/api/v1/collections/{cid}/add",
                json={"ids": ids, "documents": docs, "metadatas": metas},
            ).raise_for_status()
            logger.info(f"已从 SQLite 迁移 {len(rows)} 条历史事实到 ChromaDB {CHROMA_COLLECTION}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"历史事实迁移 ChromaDB 失败（忽略）: {e}")

    def _migrate_chroma_metadata(self) -> None:
        """Backfill owner/status metadata on old Chroma facts without exposing them."""
        if self._chroma is None:
            return
        try:
            base, cid, http = self._chroma["base"], self._chroma["cid"], self._chroma["http"]
            got = http.post(f"{base}/api/v1/collections/{cid}/get", json={"include": ["metadatas"]}).json()
            ids, metas = got.get("ids") or [], got.get("metadatas") or []
            update_ids, update_metas = [], []
            for fid, meta in zip(ids, metas):
                meta = dict(meta or {})
                changed = False
                if not meta.get("username"):
                    meta["username"] = username_from_session(meta.get("session_id", "")); changed = True
                for key, default in (("status", "active"), ("confidence", 1.0), ("expires_at", ""), ("memory_key", "")):
                    if key not in meta:
                        meta[key] = default; changed = True
                if changed:
                    update_ids.append(fid); update_metas.append(meta)
            if update_ids:
                http.post(f"{base}/api/v1/collections/{cid}/update",
                          json={"ids": update_ids, "metadatas": update_metas}).raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ChromaDB 旧事实元数据迁移失败（旧事实保持隔离）: {e}")

    # ---------- L3 会话索引（SQLite） ----------
    def save_session(self, session_id: str, title: str = "", summary: str = "") -> None:
        now = _now()
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions(id, title, summary, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, title, summary, now, now),
        )
        self._conn.commit()

    def list_sessions(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, title, summary, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "title": r[1], "summary": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in cur.fetchall()
        ]

    # ---------- L2 事实（ChromaDB 主存储，SQLite 降级） ----------
    def append_fact(
        self,
        session_id: str,
        content: str,
        category: str = "finding",
        tags: Optional[list[str]] = None,
        *,
        username: str | None = None,
        memory_key: str = "",
        confidence: float = 1.0,
        ttl_days: int | None = None,
    ) -> int:
        """Insert/update one durable fact with ownership, dedup and lifecycle metadata.

        Exact duplicates return the existing id. Supplying the same memory_key
        supersedes the previous active version. Detectable contradictory facts
        are retained with status=conflicted so the generator can disclose both.
        """
        content = (content or "").strip()
        if not content:
            return 0
        now = _now()
        owner = username or username_from_session(session_id)
        digest = _content_hash(content)
        confidence = max(0.0, min(1.0, float(confidence)))
        expires_at = (
            (datetime.now(timezone.utc) + timedelta(days=max(1, ttl_days))).isoformat(timespec="seconds")
            if ttl_days else ""
        )
        tags_csv = ",".join(tags or [])

        duplicate = self._conn.execute(
            "SELECT id FROM facts WHERE username=? AND content_hash=? AND category=? "
            "AND status IN ('active','conflicted') LIMIT 1",
            (owner, digest, category),
        ).fetchone()
        if duplicate:
            return int(duplicate[0])

        supersedes_id = None
        if memory_key:
            previous = self._conn.execute(
                "SELECT id FROM facts WHERE username=? AND memory_key=? "
                "AND status IN ('active','conflicted') ORDER BY id DESC LIMIT 1",
                (owner, memory_key),
            ).fetchone()
            if previous:
                supersedes_id = int(previous[0])
                self._set_fact_status(supersedes_id, "superseded")

        status = "active"
        candidates = self._conn.execute(
            "SELECT id, content FROM facts WHERE username=? AND category=? "
            "AND status IN ('active','conflicted') ORDER BY id DESC LIMIT 30",
            (owner, category),
        ).fetchall()
        try:
            from app.rag.evidence import analyze_evidence
            for existing_id, existing_content in candidates:
                report = analyze_evidence([
                    {"source": f"fact-{existing_id}", "text": existing_content},
                    {"source": "new-fact", "text": content},
                ])
                if report["has_conflict"]:
                    status = "conflicted"
                    self._set_fact_status(int(existing_id), "conflicted")
        except Exception:  # noqa: BLE001
            pass

        cur = self._conn.execute(
            "INSERT INTO facts(session_id, category, content, tags, created_at, username, content_hash, "
            "memory_key, status, confidence, expires_at, supersedes_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, category, content, tags_csv, now, owner, digest, memory_key,
             status, confidence, expires_at, supersedes_id),
        )
        rowid = int(cur.lastrowid)
        if self._fts:
            self._conn.execute("INSERT INTO facts_fts(rowid, content) VALUES (?, ?)", (rowid, content))
        self._conn.commit()

        # SQLite is the metadata registry even when ChromaDB is the retrieval path.
        if self._chroma is not None and self._embedding is not None:
            try:
                vec = self._embedding.embed([content])[0]
                cid = f"fact_{rowid}"
                self._chroma["http"].post(
                    f"{self._chroma['base']}/api/v1/collections/{self._chroma['cid']}/add",
                    json={
                        "ids": [cid],
                        "embeddings": [vec],
                        "documents": [content],
                        "metadatas": [
                            {
                                "session_id": session_id,
                                "category": category,
                                "tags": tags_csv,
                                "created_at": now,
                                "username": owner,
                                "memory_key": memory_key,
                                "status": status,
                                "confidence": confidence,
                                "expires_at": expires_at,
                                "supersedes_id": supersedes_id or 0,
                            }
                        ],
                    },
                ).raise_for_status()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ChromaDB 写入失败，事实已安全保存在 SQLite: {e}")
        return rowid

    def _set_fact_status(self, rowid: int, status: str) -> None:
        self._conn.execute("UPDATE facts SET status=? WHERE id=?", (status, rowid))
        if self._chroma is not None:
            try:
                row = self._conn.execute(
                    "SELECT session_id, category, tags, created_at, username, memory_key, status, confidence, expires_at, supersedes_id "
                    "FROM facts WHERE id=?", (rowid,),
                ).fetchone()
                if not row:
                    return
                sid, category, tags, created, owner, key, fact_status, confidence, expires, supersedes = row
                self._chroma["http"].post(
                    f"{self._chroma['base']}/api/v1/collections/{self._chroma['cid']}/update",
                    json={"ids": [f"fact_{rowid}"], "metadatas": [{
                        "session_id": sid, "category": category, "tags": tags or "",
                        "created_at": created, "username": owner, "memory_key": key or "",
                        "status": fact_status, "confidence": float(confidence or 0),
                        "expires_at": expires or "", "supersedes_id": supersedes or 0,
                    }]},
                )
            except Exception:  # noqa: BLE001
                pass

    def prune_expired(self, username: str | None = None) -> int:
        """Mark elapsed facts expired; returns the number transitioned."""
        sql = "SELECT id FROM facts WHERE expires_at<>'' AND expires_at<=? AND status IN ('active','conflicted')"
        params: list = [_now()]
        if username:
            sql += " AND username=?"; params.append(username)
        ids = [int(row[0]) for row in self._conn.execute(sql, params).fetchall()]
        for rowid in ids:
            self._set_fact_status(rowid, "expired")
        self._conn.commit()
        return len(ids)

    def search_facts(
        self, query: str, k: int = 5, category: Optional[str] = None,
        *, username: str | None = None, min_confidence: float = 0.0,
    ) -> list[dict]:
        """Recall facts. Production callers must pass username for tenant isolation."""
        self.prune_expired(username)
        # 主路径：ChromaDB 向量检索
        if self._chroma is not None and self._embedding is not None:
            try:
                qv = self._embedding.embed_query(query)
                body: dict = {"query_embeddings": [qv], "n_results": max(k * 4, k)}
                if username:
                    body["where"] = {"username": {"$eq": username}}
                r = self._chroma["http"].post(
                    f"{self._chroma['base']}/api/v1/collections/{self._chroma['cid']}/query",
                    json=body,
                ).json()
                ids = (r.get("ids") or [[]])[0]
                dists = (r.get("distances") or [[]])[0]
                docs = (r.get("documents") or [[]])[0]
                metas = (r.get("metadatas") or [[]])[0]
                out = []
                for i, fid in enumerate(ids):
                    m = metas[i] if i < len(metas) else {}
                    if username and m.get("username") != username:
                        continue
                    if category and m.get("category") != category:
                        continue
                    if m.get("status", "active") not in ("active", "conflicted"):
                        continue
                    if float(m.get("confidence", 1.0)) < min_confidence:
                        continue
                    out.append(
                        {
                            "id": fid,
                            "session_id": m.get("session_id", ""),
                            "category": m.get("category", ""),
                            "content": docs[i] if i < len(docs) else "",
                            "tags": (m.get("tags") or "").split(",") if m.get("tags") else [],
                            "created_at": m.get("created_at", ""),
                            "score": 1.0 - float(dists[i]) if i < len(dists) else 0.0,
                            "username": m.get("username", LEGACY_USER),
                            "status": m.get("status", "active"),
                            "confidence": float(m.get("confidence", 1.0)),
                            "expires_at": m.get("expires_at", ""),
                        }
                    )
                return out[:k]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ChromaDB 检索失败，降级 SQLite: {e}")
        # 降级路径：SQLite FTS5 / LIKE（FTS5 对中文分词不佳，命中为空时回退 LIKE）
        if self._fts:
            sql = (
                "SELECT f.id, f.session_id, f.category, f.content, f.tags, f.created_at, f.username, f.status, f.confidence, f.expires_at "
                "FROM facts_fts ft JOIN facts f ON f.id = ft.rowid "
                "WHERE facts_fts MATCH ? AND f.status IN ('active','conflicted') AND f.confidence>=?"
            )
            params: list = [query, min_confidence]
            if username:
                sql += " AND f.username = ?"; params.append(username)
            if category:
                sql += " AND f.category = ?"
                params.append(category)
            sql += " ORDER BY rank LIMIT ?"
            params.append(k)
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if not rows:
                # 中文等场景 FTS5 分词不命中 -> 回退 LIKE 子串匹配
                like_sql = (
                    "SELECT id, session_id, category, content, tags, created_at, username, status, confidence, expires_at FROM facts "
                    "WHERE content LIKE ? AND status IN ('active','conflicted') AND confidence>=?"
                )
                like_params: list = [f"%{query}%", min_confidence]
                if username:
                    like_sql += " AND username = ?"; like_params.append(username)
                if category:
                    like_sql += " AND category = ?"
                    like_params.append(category)
                like_sql += " ORDER BY id DESC LIMIT ?"
                like_params.append(k)
                rows = self._conn.execute(like_sql, like_params).fetchall()
            return [
                {
                    "id": r[0],
                    "session_id": r[1],
                    "category": r[2],
                    "content": r[3],
                    "tags": r[4].split(",") if r[4] else [],
                    "created_at": r[5],
                    "username": r[6], "status": r[7], "confidence": r[8], "expires_at": r[9],
                }
                for r in rows
            ]
        else:
            sql = (
                "SELECT id, session_id, category, content, tags, created_at, username, status, confidence, expires_at FROM facts "
                "WHERE content LIKE ? AND status IN ('active','conflicted') AND confidence>=?"
            )
            params = [f"%{query}%", min_confidence]
            if username:
                sql += " AND username = ?"; params.append(username)
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(k)
        cur = self._conn.execute(sql, params)
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "category": r[2],
                "content": r[3],
                "tags": r[4].split(",") if r[4] else [],
                "created_at": r[5],
                "username": r[6], "status": r[7], "confidence": r[8], "expires_at": r[9],
            }
            for r in cur.fetchall()
        ]

    def get_session_facts(self, session_id: str) -> list[dict]:
        """某会话沉淀的全部事实（ChromaDB 按 session_id 过滤，降级走 SQLite）。"""
        if self._chroma is not None:
            try:
                r = self._chroma["http"].post(
                    f"{self._chroma['base']}/api/v1/collections/{self._chroma['cid']}/get",
                    json={
                        "where": {"session_id": {"$eq": session_id}},
                        "include": ["documents", "metadatas"],
                    },
                ).json()
                ids = r.get("ids") or []
                docs = r.get("documents") or []
                metas = r.get("metadatas") or []
                out = []
                for i, fid in enumerate(ids):
                    m = metas[i] if i < len(metas) else {}
                    out.append(
                        {
                            "id": fid,
                            "category": m.get("category", ""),
                            "content": docs[i] if i < len(docs) else "",
                            "tags": (m.get("tags") or "").split(",") if m.get("tags") else [],
                            "created_at": m.get("created_at", ""),
                        }
                    )
                return sorted(out, key=lambda x: x["created_at"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ChromaDB 读取失败，降级 SQLite: {e}")
        cur = self._conn.execute(
            "SELECT id, category, content, tags, created_at FROM facts "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        return [
            {"id": r[0], "category": r[1], "content": r[2], "tags": r[3].split(",") if r[3] else [], "created_at": r[4]}
            for r in cur.fetchall()
        ]

    def close(self) -> None:
        if self._chroma is not None:
            try:
                self._chroma["http"].close()
            except Exception:  # noqa: BLE001
                pass
        self._conn.close()
