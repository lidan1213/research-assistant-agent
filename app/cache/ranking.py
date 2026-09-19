"""Redis 排行榜（ZSET）：检索热词 + 文档命中排行。

面试价值点：
- Sorted Set 天然排序，ZINCRBY 原子自增，适合"热搜榜/命中榜"类实时排行；
- 与 SQLite 离线聚合互补：Redis 给实时看板，SQLite 给历史统计；
- 优雅降级：Redis 不可用时静默跳过，不影响检索主流程。

用法：
    from app.cache.ranking import rank_incr, rank_top
    await rank_incr("hot_query", query)          # 检索时自增
    top = await rank_top("hot_query", 10)        # 取前 10
"""
from __future__ import annotations

from app.cache.redis_client import get_redis

_TTL = 7 * 86400  # 排行数据保留 7 天


async def rank_incr(kind: str, member: str, score: float = 1.0, username: str = "") -> None:
    """成员分数自增（如查询热词 +1、文档命中 +1）。"""
    try:
        r = get_redis()
        if r is None:
            return
        key = f"rank:{kind}:{username or 'global'}"
        pipe = r.pipeline(transaction=True)
        pipe.zincrby(key, score, member)
        pipe.expire(key, _TTL)
        await pipe.execute()
    except Exception:  # noqa: BLE001
        pass


async def rank_top(kind: str, n: int = 10, username: str = "") -> list[dict]:
    """取排行榜前 n 名：[{"member": ..., "score": ...}]，按分数降序。"""
    try:
        r = get_redis()
        if r is None:
            return []
        key = f"rank:{kind}:{username or 'global'}"
        items = await r.zrevrange(key, 0, n - 1, withscores=True)
        return [{"member": m, "score": round(float(s), 3)} for m, s in items]
    except Exception:  # noqa: BLE001
        return []


async def rank_count(kind: str, username: str = "") -> int:
    try:
        r = get_redis()
        if r is None:
            return 0
        key = f"rank:{kind}:{username or 'global'}"
        return int(await r.zcard(key))
    except Exception:  # noqa: BLE001
        return 0
