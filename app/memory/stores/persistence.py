"""Persistent conversation backends (Redis and SQLite).

- `InMemoryStore`：进程内字典，适合单机/开发。
- `RedisStore`：基于 redis.asyncio，支持多实例共享与 TTL 过期。
- `SQLiteStore`：本地文件落盘，重启不丢、零额外依赖；适合单机持久化。
- `ConversationMemory`：对外统一接口，自动截断为最近 N 轮，保持上下文不过载。
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime

from app.core.logging import get_logger
from app.llm.base import ChatMessage, MessageRole, ToolCall
from app.memory.stores.base import BaseStore

logger = get_logger("memory")


class RedisStore(BaseStore):
    """Redis 会话记忆后端：List 存消息（TTL 过期），Hash 存标题元数据。

    双写架构（L1/L2 缓存模式，面试价值点）：
    - 写入：消息同时写 Redis（热数据，读快）与 SQLite（持久化兜底）；
    - 读取：优先 Redis（O(1) List 读取），key 过期/被清理时自动从
      SQLite 回填重建 —— Redis 清空不影响会话历史与管理员查询；
    - 会话列表/标题/归档等元数据查询始终走 SQLite（chat.py 直接使用
      SQLiteStore），保证功能不因 Redis 状态而退化。
    """

    def __init__(self, url: str, ttl: int, persist_path: str = "") -> None:
        import redis.asyncio as aioredis

        self._r = aioredis.from_url(url, decode_responses=True)
        self._ttl = ttl
        # 持久化兜底：Redis 数据丢失时从 SQLite 恢复
        self._sqlite: SQLiteStore | None = None
        if persist_path:
            try:
                self._sqlite = SQLiteStore(persist_path)
            except Exception:  # noqa: BLE001
                self._sqlite = None

    def _key(self, sid: str) -> str:
        return f"agent:memory:{sid}"

    def _title_key(self, sid: str) -> str:
        return f"agent:title:{sid}"

    async def append(self, session_id: str, msg: ChatMessage) -> None:
        key = self._key(session_id)
        # 双写：Redis 热数据 + SQLite 持久化（任一失败不影响另一层）
        try:
            pipe = self._r.pipeline(transaction=True)
            pipe.rpush(key, json.dumps(msg.to_dict(), ensure_ascii=False))
            pipe.expire(key, self._ttl)
            await pipe.execute()
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                await self._sqlite.append(session_id, msg)
            except Exception:  # noqa: BLE001
                pass

    async def get(self, session_id: str) -> list[ChatMessage]:
        try:
            raw = await asyncio.wait_for(
                self._r.lrange(self._key(session_id), 0, -1), timeout=0.5
            )
        except Exception:  # noqa: BLE001
            raw = []
        if raw:
            return [ChatMessage.from_dict(json.loads(item)) for item in raw]
        # Redis 未命中：尝试从 SQLite 回填（缓存重建）
        if self._sqlite is not None:
            try:
                msgs = await self._sqlite.get(session_id)
                if msgs:
                    pipe = self._r.pipeline(transaction=True)
                    for m in msgs:
                        pipe.rpush(self._key(session_id), json.dumps(m.to_dict(), ensure_ascii=False))
                    pipe.expire(self._key(session_id), self._ttl)
                    try:
                        await asyncio.wait_for(pipe.execute(), timeout=0.5)
                    except Exception:  # noqa: BLE001
                        pass
                return msgs
            except Exception:  # noqa: BLE001
                return []
        return []

    async def clear(self, session_id: str) -> None:
        try:
            await self._r.delete(self._key(session_id))
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                await self._sqlite.clear(session_id)
            except Exception:  # noqa: BLE001
                pass

    # ---------- 标题元数据（Hash，随消息一起 TTL） ----------
    async def get_title(self, session_id: str) -> str | None:
        try:
            val = await self._r.hget(self._title_key(session_id), "title")
            if val:
                return val
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                return await asyncio.to_thread(self._sqlite.get_title, session_id)
            except Exception:  # noqa: BLE001
                return None
        return None

    async def set_title(self, session_id: str, title: str) -> None:
        try:
            pipe = self._r.pipeline(transaction=True)
            pipe.hset(self._title_key(session_id), "title", title)
            pipe.expire(self._title_key(session_id), self._ttl)
            await pipe.execute()
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                await asyncio.to_thread(self._sqlite.set_title, session_id, title)
            except Exception:  # noqa: BLE001
                pass

    async def get_summary(self, session_id: str) -> str | None:
        try:
            val = await self._r.hget(self._title_key(session_id), "summary")
            if val:
                return val
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                return await asyncio.to_thread(self._sqlite.get_summary, session_id)
            except Exception:  # noqa: BLE001
                return None
        return None

    async def set_summary(self, session_id: str, summary: str) -> None:
        try:
            pipe = self._r.pipeline(transaction=True)
            pipe.hset(self._title_key(session_id), "summary", summary)
            pipe.expire(self._title_key(session_id), self._ttl)
            await pipe.execute()
        except Exception:  # noqa: BLE001
            pass
        if self._sqlite is not None:
            try:
                await asyncio.to_thread(self._sqlite.set_summary, session_id, summary)
            except Exception:  # noqa: BLE001
                pass


class SQLiteStore(BaseStore):
    """本地文件落盘的会话记忆后端（零额外依赖，重启不丢）。"""

    def __init__(self, path: str = "./data/memory.db") -> None:
        self._path = path
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, tool_call_id TEXT, name TEXT, tool_calls TEXT)"
        )
        # 会话标题表（自动生成的对话标题）
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_titles("
            "session_id TEXT PRIMARY KEY, title TEXT, summary TEXT, updated_at TEXT)"
        )
        # 旧表迁移：为已存在的表补充 tool_calls 列（幂等）
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(messages)")]
        if "tool_calls" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
        # 旧表迁移：session_titles 补充 summary 列（幂等）
        tcols = [r[1] for r in self._conn.execute("PRAGMA table_info(session_titles)")]
        if "summary" not in tcols:
            self._conn.execute("ALTER TABLE session_titles ADD COLUMN summary TEXT")
        # 旧表迁移：session_titles 补充 archived 列（会话归档标记，幂等）
        if "archived" not in tcols:
            self._conn.execute("ALTER TABLE session_titles ADD COLUMN archived INTEGER DEFAULT 0")
        self._conn.commit()

    # ---------- 会话标题 ----------
    def set_title(self, session_id: str, title: str) -> None:
        """保存/更新会话标题（幂等 upsert，保留已有 summary）。"""
        self._conn.execute(
            "INSERT INTO session_titles(session_id, title, summary, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET title=excluded.title, "
            "updated_at=excluded.updated_at",
            (session_id, title, self.get_summary(session_id) or "",
             datetime.now().isoformat(timespec="seconds")),
        )
        self._conn.commit()

    def get_title(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT title FROM session_titles WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def set_summary(self, session_id: str, summary: str) -> None:
        """保存/更新会话摘要（幂等 upsert）。"""
        self._conn.execute(
            "INSERT INTO session_titles(session_id, title, summary, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "updated_at=excluded.updated_at",
            (session_id, self.get_title(session_id) or "", summary,
             datetime.now().isoformat(timespec="seconds")),
        )
        self._conn.commit()

    def get_summary(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT summary FROM session_titles WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def get_archived(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT archived FROM session_titles WHERE session_id=?", (session_id,)
        ).fetchone()
        return bool(row and row[0])

    def set_archived(self, session_id: str, archived: bool = True) -> None:
        """标记/取消会话归档（归档会话从前端列表隐藏，搜索仍可命中）。"""
        self._conn.execute(
            "INSERT INTO session_titles(session_id, title, summary, archived, updated_at) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET archived=excluded.archived",
            (
                session_id,
                self.get_title(session_id) or "",
                self.get_summary(session_id) or "",
                1 if archived else 0,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()

    def archive_old_sessions(self, prefix: str = "", days: int = 30) -> int:
        """把超过 days 天未活跃（updated_at 早于阈值）的会话标记为归档。返回归档数。"""
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        sql = (
            "UPDATE session_titles SET archived=1 "
            "WHERE (archived IS NULL OR archived=0) AND updated_at < ?"
        )
        params: list = [cutoff]
        if prefix:
            sql += " AND session_id LIKE ?"
            params.append(f"{prefix}%")
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    def unarchive_all(self, prefix: str = "") -> int:
        """取消所有（或指定前缀下）会话的归档标记。返回恢复数。"""
        sql = "UPDATE session_titles SET archived=0 WHERE archived=1"
        params: list = []
        if prefix:
            sql += " AND session_id LIKE ?"
            params.append(f"{prefix}%")
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _dump_tool_calls(msg: ChatMessage) -> str | None:
        if not msg.tool_calls:
            return None
        return json.dumps(
            [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ],
            ensure_ascii=False,
        )

    @staticmethod
    def _load_tool_calls(raw: str | None) -> list | None:
        if not raw:
            return None
        try:
            return [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in json.loads(raw)
            ]
        except Exception:  # noqa: BLE001
            return None

    def _row_to_msg(self, row) -> ChatMessage:
        return ChatMessage(
            role=MessageRole(row[1]),
            content=row[2] or "",
            tool_call_id=row[3],
            name=row[4],
            tool_calls=self._load_tool_calls(row[5]) if len(row) > 5 else None,
        )

    async def append(self, session_id: str, msg: ChatMessage) -> None:
        self._conn.execute(
            "INSERT INTO messages(session_id, role, content, tool_call_id, name, tool_calls) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                msg.role.value,
                msg.content,
                msg.tool_call_id,
                msg.name,
                self._dump_tool_calls(msg),
            ),
        )
        self._conn.commit()

    async def get(self, session_id: str) -> list[ChatMessage]:
        cur = self._conn.execute(
            "SELECT id, role, content, tool_call_id, name, tool_calls FROM messages "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        return [self._row_to_msg(r) for r in cur.fetchall()]

    def list_sessions(self, prefix: str = "") -> list[dict]:
        """列出会话（可按 session_id 前缀过滤，如按用户名 'student1__'）。

        返回 [{session_id, message_count, first_at, last_at}]，按最后活动时间倒序。
        """
        if prefix:
            cur = self._conn.execute(
                "SELECT session_id, COUNT(*), MIN(id), MAX(id) FROM messages "
                "WHERE session_id LIKE ? GROUP BY session_id",
                (f"{prefix}%",),
            )
        else:
            cur = self._conn.execute(
                "SELECT session_id, COUNT(*), MIN(id), MAX(id) FROM messages "
                "GROUP BY session_id"
            )
        out = []
        for sid, cnt, min_id, max_id in cur.fetchall():
            first = self._conn.execute(
                "SELECT content FROM messages WHERE id=?", (min_id,)
            ).fetchone()
            last = self._conn.execute(
                "SELECT content FROM messages WHERE id=?", (max_id,)
            ).fetchone()
            out.append(
                {
                    "session_id": sid,
                    "message_count": cnt,
                    "first_message": (first[0] or "")[:100] if first else "",
                    "last_message": (last[0] or "")[:100] if last else "",
                    "title": self.get_title(sid) or "",  # 自动生成的会话标题
                    "summary": self.get_summary(sid) or "",  # 会话自动摘要（可为空）
                    "archived": self.get_archived(sid),  # 归档标记（列表隐藏，搜索可查）
                }
            )
        # 按最大消息 id（最后活动）倒序
        out.sort(key=lambda x: -self._max_id(x["session_id"]))
        return out

    def statistics(self) -> dict[str, int]:
        """Return aggregate counts without exposing the SQLite connection."""
        session_count = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages"
        ).fetchone()[0]
        message_count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {"sessions": session_count, "messages": message_count}

    def _max_id(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0] or 0

    def truncate_to_last_user(self, session_id: str) -> int:
        """删除最后一个 user 提问之后的所有消息（重试前清掉半成品回答）。

        返回删除条数；会话不存在或没有 user 消息时返回 0。
        """
        row = self._conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id=? AND role='user'",
            (session_id,),
        ).fetchone()
        last_user_id = row[0]
        if last_user_id is None:
            return 0
        cur = self._conn.execute(
            "DELETE FROM messages WHERE session_id=? AND id>?",
            (session_id, last_user_id),
        )
        self._conn.commit()
        return cur.rowcount

    def search_messages(self, q: str, prefix: str = "", limit: int = 30) -> list[dict]:
        """会话全文搜索：LIKE 子串匹配（对中文比 FTS5 分词更稳）。

        返回命中会话列表 [{session_id, title, hits, snippet}]，按最近命中倒序。
        prefix 传 "{username}__" 限定某用户，传 "" 搜索全部。
        """
        like = f"%{q}%"
        sql = (
            "SELECT m.session_id, COUNT(*) AS hits, MAX(m.id) AS last_id "
            "FROM messages m "
            "WHERE m.content LIKE ? AND m.role IN ('user','assistant')"
        )
        params: list = [like]
        if prefix:
            sql += " AND m.session_id LIKE ?"
            params.append(f"{prefix}%")
        sql += " GROUP BY m.session_id ORDER BY last_id DESC LIMIT ?"
        params.append(limit)
        out = []
        try:
            rows = self._conn.execute(sql, params).fetchall()
            for sid, hits, last_id in rows:
                snippet_row = self._conn.execute(
                    "SELECT content FROM messages WHERE session_id=? AND content LIKE ? "
                    "ORDER BY id DESC LIMIT 1",
                    (sid, like),
                ).fetchone()
                snippet = (snippet_row[0] or "").strip()[:80] if snippet_row else ""
                out.append(
                    {
                        "session_id": sid,
                        "title": self.get_title(sid) or sid,
                        "hits": hits,
                        "snippet": snippet,
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"会话全文搜索失败: {e}")
        return out

    def prune_sessions(self, prefix: str = "", keep: int = 50) -> int:
        """清理指定前缀下的历史会话，只保留最近 keep 个（按最后活动倒序）。

        返回删除的会话数。keep<=0 表示不清理。
        """
        if keep <= 0:
            return 0
        sessions = self.list_sessions(prefix=prefix)
        if len(sessions) <= keep:
            return 0
        deleted = 0
        for s in sessions[keep:]:  # 超出保留数的（最旧）逐会话删除
            sid = s["session_id"]
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            self._conn.execute(
                "DELETE FROM session_titles WHERE session_id=?", (sid,)
            )
            deleted += 1
        self._conn.commit()
        return deleted

    async def clear(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        """彻底删除一个会话（消息 + 标题/摘要/归档标记）。"""
        self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM session_titles WHERE session_id=?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
