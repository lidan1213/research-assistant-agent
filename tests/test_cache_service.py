"""CacheService 统一缓存层测试：hit/miss/fallback 计数 + fail-open 降级。"""
from __future__ import annotations

import asyncio

import pytest

from app.memory.cache import CacheService, get_cache_service, reset_cache_service


class FakeRedis:
    def __init__(self):
        self._data: dict = {}

    async def get(self, key):
        return self._data.get(key)

    async def set(self, key, val, ex=None):
        self._data[key] = val

    async def delete(self, *keys):
        for k in keys:
            self._data.pop(k, None)

    async def scan_iter(self, match="*"):
        import fnmatch

        for k in list(self._data):
            if fnmatch.fnmatch(k, match):
                yield k


@pytest.fixture()
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("app.cache.redis_client._pool", r)
    monkeypatch.setattr("app.cache.redis_client._disabled", False)
    return r


def test_cache_get_set(fake_redis):
    svc = CacheService()
    async def run():
        await svc.set("retrieve", "q1", {"hits": ["a"]}, "u1")
        assert await svc.get("retrieve", "q1", "u1") == {"hits": ["a"]}
        assert await svc.get("retrieve", "q1", "u2") is None  # 用户隔离

    asyncio.run(run())
    assert svc.stats["hit"] == 1
    assert svc.stats["miss"] == 1


def test_cache_delete_and_invalidate(fake_redis):
    svc = CacheService()
    async def run():
        await svc.set("retrieve", "k1", [1], "u")
        await svc.delete("retrieve", "k1", "u")
        assert await svc.get("retrieve", "k1", "u") is None
        await svc.set("retrieve", "k2", [2], "u")
        await svc.invalidate_user("u")
        assert await svc.get("retrieve", "k2", "u") is None

    asyncio.run(run())


def test_cache_get_or_set_compute(fake_redis):
    svc = CacheService()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return {"data": "expensive"}

    async def run():
        v1 = await svc.get_or_set("retrieve", "q", compute, "u")
        v2 = await svc.get_or_set("retrieve", "q", compute, "u")
        assert v1 == v2 == {"data": "expensive"}

    asyncio.run(run())
    assert calls["n"] == 1  # 第二次命中缓存，compute 未再调用


def test_cache_get_or_set_empty_short_ttl(fake_redis):
    """空结果也缓存（短 TTL 防穿透），第二次不重复计算。"""
    svc = CacheService()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return []

    async def run():
        await svc.get_or_set("retrieve", "q", compute, "u")
        await svc.get_or_set("retrieve", "q", compute, "u")

    asyncio.run(run())
    assert calls["n"] == 1


def test_cache_fail_open(monkeypatch):
    """Redis 不可用（_pool=None）时：get 返回 None，set 不抛，计数 fallback。"""
    monkeypatch.setattr("app.cache.redis_client._pool", None)
    monkeypatch.setattr("app.cache.redis_client._disabled", True)
    svc = CacheService()

    async def run():
        assert await svc.get("retrieve", "q", "u") is None
        await svc.set("retrieve", "q", [1], "u")  # 不抛
        await svc.invalidate_user("u")  # 不抛
        return True

    assert asyncio.run(run())
    assert svc.stats["fallback"] >= 1


def test_retrieval_cache_delegates_to_service(monkeypatch):
    """旧接口（cache_get/cache_set）委托 CacheService，行为一致。"""
    r = FakeRedis()
    monkeypatch.setattr("app.cache.redis_client._pool", r)
    monkeypatch.setattr("app.cache.redis_client._disabled", False)
    reset_cache_service()

    from app.cache.retrieval_cache import cache_get, cache_set

    async def run():
        await cache_set("retrieve", "k", {"v": 1}, "u")
        return await cache_get("retrieve", "k", "u")

    assert asyncio.run(run()) == {"v": 1}
    reset_cache_service()
