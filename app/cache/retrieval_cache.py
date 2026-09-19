"""Redis 检索缓存（兼容层）：委托统一 CacheService（app.memory.cache）。

本模块保留原函数签名（cache_get/cache_set/cache_get_or_set/invalidate_user_kb），
现有调用方（retriever / user_kb / notes / kg）零改动；内部实现已迁移到
CacheService——所有 Redis 异常在服务层 fail-open，业务层无 try/except。

面试价值点：
- 缓存降本：高频相似问题直接命中缓存，省 embedding + 检索 + 重排；
- TTL 一致性：知识库文档变更时主动失效（invalidate_user_kb）；
- 穿透保护：空结果也缓存短 TTL；
- 优雅降级：Redis 不可用时自动跳过缓存，主链路不受影响。
"""
from __future__ import annotations

from typing import Any

from app.memory.cache import get_cache_service


async def cache_get(kind: str, raw: str, username: str = "") -> Any | None:
    return await get_cache_service().get(kind, raw, username)


async def cache_set(
    kind: str, raw: str, data: Any, username: str = "", ttl: int = 300
) -> None:
    await get_cache_service().set(kind, raw, data, username, ttl)


async def cache_delete_prefix(kind: str) -> None:
    await get_cache_service().invalidate_kind(kind)


async def invalidate_user_kb(username: str) -> None:
    await get_cache_service().invalidate_user(username)


async def cache_get_or_set(
    kind: str,
    raw: str,
    compute,
    username: str = "",
    ttl: int = 300,
    empty_ttl: int = 60,
) -> Any:
    return await get_cache_service().get_or_set(
        kind, raw, compute, username, ttl, empty_ttl
    )
