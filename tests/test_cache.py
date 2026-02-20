"""Unit tests for the QueryCache (async-safe LRU with TTL)."""

from __future__ import annotations

import time

import pytest
from src.application.cache import QueryCache


class TestQueryCacheBasics:
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        await cache.put("What is Article 5?", {"answer": "Prohibitions"})
        result = await cache.get("What is Article 5?")
        assert result is not None
        assert result["answer"] == "Prohibitions"

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        assert await cache.get("unknown question") is None

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        await cache.put("What Is Article 5?", {"answer": "test"})
        assert await cache.get("what is article 5?") is not None
        assert await cache.get("WHAT IS ARTICLE 5?") is not None

    @pytest.mark.asyncio
    async def test_whitespace_normalized(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        await cache.put("  What is Article 5?  ", {"answer": "test"})
        assert await cache.get("What is Article 5?") is not None

    @pytest.mark.asyncio
    async def test_size_property(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        assert cache.size == 0
        await cache.put("q1", {"a": 1})
        assert cache.size == 1
        await cache.put("q2", {"a": 2})
        assert cache.size == 2


class TestQueryCacheTTL:
    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self):
        cache = QueryCache(max_size=10, ttl_seconds=0.1)
        await cache.put("q", {"answer": "old"})
        time.sleep(0.15)
        assert await cache.get("q") is None

    @pytest.mark.asyncio
    async def test_fresh_entry_returns_value(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        await cache.put("q", {"answer": "fresh"})
        assert await cache.get("q") is not None


class TestQueryCacheLRUEviction:
    @pytest.mark.asyncio
    async def test_evicts_oldest_when_full(self):
        cache = QueryCache(max_size=3, ttl_seconds=60.0)
        await cache.put("q1", {"a": 1})
        await cache.put("q2", {"a": 2})
        await cache.put("q3", {"a": 3})
        # q1 is the oldest
        await cache.put("q4", {"a": 4})
        assert await cache.get("q1") is None  # evicted
        assert await cache.get("q2") is not None
        assert await cache.get("q4") is not None
        assert cache.size == 3

    @pytest.mark.asyncio
    async def test_access_refreshes_lru_order(self):
        cache = QueryCache(max_size=3, ttl_seconds=60.0)
        await cache.put("q1", {"a": 1})
        await cache.put("q2", {"a": 2})
        await cache.put("q3", {"a": 3})
        # Access q1 to move it to most-recent
        await cache.get("q1")
        # Now q2 is the oldest
        await cache.put("q4", {"a": 4})
        assert await cache.get("q2") is None  # evicted
        assert await cache.get("q1") is not None  # survived because accessed


class TestQueryCacheOverwrite:
    @pytest.mark.asyncio
    async def test_overwrite_updates_value(self):
        cache = QueryCache(max_size=10, ttl_seconds=60.0)
        await cache.put("q", {"answer": "old"})
        await cache.put("q", {"answer": "new"})
        result = await cache.get("q")
        assert result is not None
        assert result["answer"] == "new"
        assert cache.size == 1
