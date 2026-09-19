"""Redis 缓存/限流/排行模块测试。

策略：mock redis.asyncio 客户端（不依赖真实 Redis 服务），
验证逻辑正确性与「Redis 不可用时优雅降级」两条关键路径。
"""
from __future__ import annotations

import asyncio

import pytest

from app.cache.ratelimit import fixed_window, sliding_window
from app.cache.ranking import rank_incr, rank_top
from app.cache.retrieval_cache import cache_get, cache_set, invalidate_user_kb


# ---------- 优雅降级：get_redis 返回 None（模拟 Redis 挂掉） ----------
def test_redis_disabled_fail_open(monkeypatch):
    """Redis 不可用时所有模块必须 fail-open，绝不抛异常。"""
    monkeypatch.setattr("app.cache.redis_client._pool", None)
    monkeypatch.setattr("app.cache.redis_client._disabled", True)

    async def run():
        assert await cache_get("retrieve", "q", "u") is None
        await cache_set("retrieve", "q", [1, 2], "u")  # 不应抛
        ok, retry = await fixed_window("chat", "u", limit=5, window=60)
        assert ok is True and retry == 0
        ok, retry = await sliding_window("chat", "u", limit=5, window=60)
        assert ok is True and retry == 0
        await rank_incr("hot_query", "钙钛矿")  # 不应抛
        assert await rank_top("hot_query", 5) == []
        await invalidate_user_kb("u")  # 不应抛

    asyncio.run(run())


# ---------- mock Redis 客户端 ----------
class FakeRedis:
    """最小 Redis mock：支持 get/set/incr/expire/ttl/zadd/zincrby/zrevrange/zcard/pipeline。"""

    def __init__(self):
        self._data: dict = {}
        self._ttl: dict = {}
        self._z: dict[str, dict] = {}

    async def get(self, key):
        return self._data.get(key)

    async def set(self, key, val, ex=None):
        self._data[key] = val
        if ex:
            self._ttl[key] = ex

    async def incr(self, key):
        self._data[key] = int(self._data.get(key, 0)) + 1
        return self._data[key]

    async def ttl(self, key):
        return self._ttl.get(key, -1)

    async def expire(self, key, ttl):
        self._ttl[key] = ttl

    async def delete(self, *keys):
        for k in keys:
            self._data.pop(k, None)
            self._z.pop(k, None)

    async def zadd(self, key, mapping):
        self._z.setdefault(key, {}).update(mapping)

    async def zincrby(self, key, score, member):
        self._z.setdefault(key, {})
        self._z[key][member] = float(self._z[key].get(member, 0)) + score

    async def zrevrange(self, key, start, end, withscores=False):
        items = sorted(self._z.get(key, {}).items(), key=lambda x: x[1], reverse=True)
        items = items[start : end + 1]
        if withscores:
            return [(m, s) for m, s in items]
        return [m for m, _ in items]

    async def zcard(self, key):
        return len(self._z.get(key, {}))

    async def zremrangebyscore(self, key, mn, mx):
        self._z.setdefault(key, {})
        keep = {m: s for m, s in self._z[key].items() if not (mn <= s <= mx)}
        removed = len(self._z[key]) - len(keep)
        self._z[key] = keep
        return removed

    async def zrange(self, key, start, end, withscores=False):
        items = sorted(self._z.get(key, {}).items(), key=lambda x: x[1])
        items = items[start : end + 1]
        if withscores:
            return [(m, s) for m, s in items]
        return [m for m, _ in items]

    async def scan_iter(self, match="*"):
        import fnmatch

        for k in list(self._data) + list(self._z):
            if fnmatch.fnmatch(k, match):
                yield k

    async def dbsize(self):
        return len(self._data) + len(self._z)

    async def keys(self, pattern="*"):
        import fnmatch

        return [k for k in list(self._data) + list(self._z) if fnmatch.fnmatch(k, pattern)]

    async def ping(self):
        return True

    def pipeline(self, transaction=True):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, r: FakeRedis):
        self._r = r
        self._cmds: list = []

    def incr(self, key):
        self._cmds.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self._cmds.append(("expire", key, ttl))
        return self

    def rpush(self, key, *vals):
        self._cmds.append(("rpush", key, vals))
        return self

    def hset(self, key, field, val):
        self._cmds.append(("hset", key, field, val))
        return self

    def zadd(self, key, mapping):
        self._cmds.append(("zadd", key, mapping))
        return self

    def zincrby(self, key, score, member):
        self._cmds.append(("zincrby", key, score, member))
        return self

    def zremrangebyscore(self, key, mn, mx):
        self._cmds.append(("zremrangebyscore", key, mn, mx))
        return self

    def zcard(self, key):
        self._cmds.append(("zcard", key))
        return self

    async def execute(self):
        results = []
        for cmd in self._cmds:
            op = cmd[0]
            if op == "incr":
                results.append(await self._r.incr(cmd[1]))
            elif op == "expire":
                await self._r.expire(cmd[1], cmd[2])
                results.append(1)
            elif op == "rpush":
                key, vals = cmd[1], cmd[2]
                self._r._data.setdefault(key, [])
                self._r._data[key].extend(vals)
                results.append(len(vals))
            elif op == "hset":
                key, field, val = cmd[1], cmd[2], cmd[3]
                self._r._data.setdefault(key, {})
                self._r._data[key][field] = val
                results.append(1)
            elif op == "zadd":
                await self._r.zadd(cmd[1], cmd[2])
                results.append(len(cmd[2]))
            elif op == "zincrby":
                await self._r.zincrby(cmd[1], cmd[2], cmd[3])
                results.append(1)
            elif op == "zremrangebyscore":
                results.append(await self._r.zremrangebyscore(cmd[1], cmd[2], cmd[3]))
            elif op == "zcard":
                results.append(await self._r.zcard(cmd[1]))
        self._cmds = []
        return results


@pytest.fixture()
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("app.cache.redis_client._pool", r)
    monkeypatch.setattr("app.cache.redis_client._disabled", False)
    return r


def test_retrieval_cache_set_get(fake_redis):
    async def run():
        await cache_set("retrieve", "钙钛矿电池", {"hits": ["doc1"]}, "student1")
        hit = await cache_get("retrieve", "钙钛矿电池", "student1")
        assert hit == {"hits": ["doc1"]}
        # 不同用户不串数据
        miss = await cache_get("retrieve", "钙钛矿电池", "student2")
        assert miss is None

    asyncio.run(run())


def test_retrieval_cache_invalidate(fake_redis):
    async def run():
        await cache_set("retrieve", "q1", [1], "student1")
        await cache_set("retrieve", "q2", [2], "student1")
        await invalidate_user_kb("student1")
        assert await cache_get("retrieve", "q1", "student1") is None
        assert await cache_get("retrieve", "q2", "student1") is None

    asyncio.run(run())


def test_fixed_window_rate_limit(fake_redis):
    async def run():
        # 限 3 次/窗口
        for _ in range(3):
            ok, _ = await fixed_window("chat", "u1", limit=3, window=60)
            assert ok is True
        ok, retry = await fixed_window("chat", "u1", limit=3, window=60)
        assert ok is False
        assert retry >= 1
        # 其他用户不受影响
        ok, _ = await fixed_window("chat", "u2", limit=3, window=60)
        assert ok is True

    asyncio.run(run())


def test_sliding_window_rate_limit(fake_redis):
    async def run():
        for _ in range(3):
            ok, _ = await sliding_window("chat", "u1", limit=3, window=60)
            assert ok is True
        ok, retry = await sliding_window("chat", "u1", limit=3, window=60)
        assert ok is False
        assert retry >= 1

    asyncio.run(run())


def test_ranking_zset(fake_redis):
    async def run():
        await rank_incr("hot_query", "钙钛矿", 1.0, "global")
        await rank_incr("hot_query", "钙钛矿", 1.0, "global")
        await rank_incr("hot_query", "Transformer", 1.0, "global")
        top = await rank_top("hot_query", 5, "global")
        assert top[0]["member"] == "钙钛矿"
        assert top[0]["score"] == 2.0
        assert len(top) == 2

    asyncio.run(run())
