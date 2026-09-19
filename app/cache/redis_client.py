"""Redis 统一客户端：懒加载连接池 + 优雅降级。

设计要点：
- 懒加载：首次使用时才建立连接（不阻塞应用启动）。
- 优雅降级：Redis 不可用时所有操作安全返回默认值/None，绝不抛异常
  打断主流程（会话记忆、检索缓存、限流在 Redis 故障时自动退化为
  SQLite / 无缓存 / 不限流），这是多组件共享基础设施的容错基线。
- 单一连接池：全局复用，避免每个调用新建连接。

用法：
    from app.cache.redis_client import get_redis, redis_available
    r = get_redis()
    if r is not None:
        await r.set("k", "v", ex=60)
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("cache.redis")

_pool: Any = None
_url: str | None = None
_disabled = False


def redis_url() -> str:
    """Redis 连接地址：环境变量 > 配置 > 默认本地。"""
    global _url
    if _url is not None:
        return _url
    env = os.environ.get("MEMORY__REDIS_URL") or os.environ.get("REDIS_URL")
    if env:
        _url = env
        return _url
    try:
        from app.config import get_settings

        _url = get_settings().memory.redis_url or "redis://127.0.0.1:6379/0"
    except Exception:  # noqa: BLE001
        _url = "redis://127.0.0.1:6379/0"
    return _url


def get_redis() -> Any | None:
    """获取全局 Redis 连接（懒加载）。失败返回 None，调用方需判空。"""
    global _pool, _disabled
    if _disabled:
        return None
    if _pool is not None:
        return _pool
    try:
        import redis.asyncio as aioredis

        _pool = aioredis.from_url(
            redis_url(),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=3,
            max_connections=20,
        )
        return _pool
    except Exception as e:  # noqa: BLE001
        _disabled = True
        logger.warning("Redis 不可用（已禁用缓存/限流，不影响主流程）: %s", e)
        return None


async def redis_available() -> bool:
    """探测 Redis 是否可用（带超时，失败自动禁用）。"""
    global _disabled
    if _disabled:
        return False
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(await r.ping())
    except Exception as e:  # noqa: BLE001
        _disabled = True
        logger.warning("Redis ping 失败，已禁用: %s", e)
        return False
