"""Redis 限流：固定窗口 + 滑动窗口，按用户/IP/接口维度限流。

面试价值点：
- 固定窗口：INCR + EXPIRE 原子实现，O(1)，适合粗粒度限流；
- 滑动窗口：ZSET 时间戳去重计数，边界更平滑，防"窗口临界突发"；
- 分布式：多实例部署时共享同一 Redis 计数，天然跨进程生效；
- 优雅降级：Redis 不可用时放行（fail-open），保证可用性优先。

用法：
    from app.cache.ratelimit import fixed_window, sliding_window
    ok, retry_after = await fixed_window("chat", user_id, limit=30, window=60)
"""
from __future__ import annotations

import time

from app.cache.redis_client import get_redis


async def fixed_window(
    bucket: str,
    identity: str,
    limit: int = 30,
    window: int = 60,
) -> tuple[bool, int]:
    """固定窗口限流：limit 次 / window 秒。

    返回 (是否放行, 建议重试等待秒数)。Redis 不可用时放行。
    """
    try:
        r = get_redis()
        if r is None:
            return True, 0
        key = f"rl:fw:{bucket}:{identity}"
        now = int(time.time())
        pipe = r.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        count = int(results[0])
        if count == 1:
            return True, 0
        if count > limit:
            ttl = await r.ttl(key)
            return False, max(ttl, 1)
        return True, 0
    except Exception:  # noqa: BLE001
        return True, 0


async def sliding_window(
    bucket: str,
    identity: str,
    limit: int = 30,
    window: int = 60,
) -> tuple[bool, int]:
    """滑动窗口限流：窗口内每个请求记录时间戳，统计最近 window 秒内次数。

    比固定窗口平滑，但每个请求多一个 ZADD + ZREMRANGEBYSCORE，开销略高。
    适合对限流精度要求高的接口（如登录/支付类）。
    """
    try:
        r = get_redis()
        if r is None:
            return True, 0
        key = f"rl:sw:{bucket}:{identity}"
        now = time.time()
        # 清理窗口外旧记录 + 写入当前时间戳（原子 Lua 更优，这里用 pipeline 简化）
        pipe = r.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        _, _, count, _ = await pipe.execute()
        if int(count) > limit:
            # 最早一条时间戳决定重试时间
            oldest = await r.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = max(int(window - (now - oldest[0][1])), 1)
            else:
                retry_after = 1
            return False, retry_after
        return True, 0
    except Exception:  # noqa: BLE001
        return True, 0
