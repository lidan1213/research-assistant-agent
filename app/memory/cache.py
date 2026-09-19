"""Cache Service：统一缓存层（业务代码不再直接调用 Redis Client）。

目标（架构重构）：
- 业务层只依赖本服务的 get/set/delete/invalidate 四个方法；
- 所有 Redis 异常在本层处理（fail-open），业务层零 try/except；
- 记录 hit / miss / timeout / fallback 计数，供运营看板统计缓存效果；
- Redis 不可用时自动降级为「无缓存」（每项操作返回 miss/None），Agent 与 RAG 主链路不受影响。

用法：
    from app.memory.cache import cache_service
    val = await cache_service.get("retrieve", "query|k=5", "student1")
    await cache_service.set("retrieve", "query|k=5", data, "student1", ttl=300)
    await cache_service.delete("retrieve", "key", "student1")
    await cache_service.invalidate_user("student1")
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.cache.redis_client import get_redis

_PREFIX = "rag"


class CacheService:
    """统一缓存门面（无状态；统计计数放内存，不影响主流程）。"""

    def __init__(self) -> None:
        self.stats = {"hit": 0, "miss": 0, "timeout": 0, "fallback": 0}

    # ---------- key 构造 ----------
    def _key(self, kind: str, raw: str, username: str = "") -> str:
        digest = hashlib.sha1(f"{username}:{kind}:{raw}".encode("utf-8")).hexdigest()[:24]
        return f"{_PREFIX}:{kind}:{digest}"

    def _prefix(self, kind: str) -> str:
        return f"{_PREFIX}:{kind}"

    # ---------- 核心操作 ----------
    async def get(self, kind: str, raw: str, username: str = "") -> Any | None:
        """读缓存；未命中或 Redis 不可用返回 None。"""
        try:
            r = get_redis()
            if r is None:
                self.stats["fallback"] += 1
                return None
            val = await r.get(self._key(kind, raw, username))
            if val is None:
                self.stats["miss"] += 1
                return None
            self.stats["hit"] += 1
            return json.loads(val)
        except Exception:  # noqa: BLE001
            self.stats["timeout"] += 1
            return None

    async def set(
        self, kind: str, raw: str, data: Any, username: str = "", ttl: int = 300
    ) -> None:
        """写缓存（失败静默）。"""
        try:
            r = get_redis()
            if r is None:
                return
            await r.set(
                self._key(kind, raw, username),
                json.dumps(data, ensure_ascii=False),
                ex=ttl,
            )
        except Exception:  # noqa: BLE001
            pass

    async def delete(self, kind: str, raw: str, username: str = "") -> None:
        try:
            r = get_redis()
            if r is None:
                return
            await r.delete(self._key(kind, raw, username))
        except Exception:  # noqa: BLE001
            pass

    async def invalidate_user(self, username: str = "") -> None:
        """按前缀批量失效（知识库变更时调用，保证一致性）。"""
        try:
            r = get_redis()
            if r is None:
                return
            keys = []
            async for k in r.scan_iter(match=f"{_PREFIX}:*"):
                keys.append(k)
            if keys:
                await r.delete(*keys)
        except Exception:  # noqa: BLE001
            pass

    async def invalidate_kind(self, kind: str) -> None:
        try:
            r = get_redis()
            if r is None:
                return
            keys = []
            async for k in r.scan_iter(match=f"{self._prefix(kind)}:*"):
                keys.append(k)
            if keys:
                await r.delete(*keys)
        except Exception:  # noqa: BLE001
            pass

    # ---------- 便捷：读-计算-回填 ----------
    async def get_or_set(
        self,
        kind: str,
        raw: str,
        compute,
        username: str = "",
        ttl: int = 300,
        empty_ttl: int = 60,
    ) -> Any:
        """读缓存，未命中调用 compute 并回填（空结果短 TTL 防穿透）。"""
        hit = await self.get(kind, raw, username)
        if hit is not None:
            return hit
        data = await compute()
        if data is None:
            return None
        is_empty = data == [] or data == {} or data is None
        await self.set(kind, raw, data, username, ttl=empty_ttl if is_empty else ttl)
        return data

    def stats_snapshot(self) -> dict:
        return dict(self.stats)


# 全局单例
_service: CacheService | None = None


def get_cache_service() -> CacheService:
    global _service
    if _service is None:
        _service = CacheService()
    return _service


def reset_cache_service() -> None:
    global _service
    _service = None


cache_service = get_cache_service()
