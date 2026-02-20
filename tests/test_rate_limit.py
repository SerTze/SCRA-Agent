"""Unit tests for the rate-limiter middleware (token bucket + TTL eviction)."""

from __future__ import annotations

import time

from src.presentation.rate_limit import RateLimitMiddleware, _TokenBucket


class TestTokenBucket:
    def test_initial_tokens_equal_burst(self):
        bucket = _TokenBucket(rate=1.0, burst=5)
        assert bucket.tokens == 5.0

    def test_consume_decrements(self):
        bucket = _TokenBucket(rate=1.0, burst=5)
        assert bucket.consume() is True
        assert bucket.tokens == 4.0

    def test_exhaust_returns_false(self):
        bucket = _TokenBucket(rate=0.0, burst=2)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_idle_seconds(self):
        bucket = _TokenBucket(rate=1.0, burst=5)
        time.sleep(0.05)
        assert bucket.idle_seconds >= 0.04


class TestRateLimitEviction:
    def test_stale_buckets_are_evicted(self):
        """Buckets idle longer than TTL should be removed by _maybe_evict."""
        middleware = RateLimitMiddleware(app=None, requests_per_minute=60, burst=10)

        # Manually insert a bucket and make it look stale
        bucket = _TokenBucket(rate=1.0, burst=10)
        bucket.last_refill = time.monotonic() - 700  # >600s TTL
        middleware._buckets["1.2.3.4"] = bucket

        # Force eviction to run (bypass interval check)
        middleware._last_eviction = 0
        middleware._maybe_evict()

        assert "1.2.3.4" not in middleware._buckets

    def test_active_buckets_are_kept(self):
        """Buckets that are still active should survive eviction."""
        middleware = RateLimitMiddleware(app=None, requests_per_minute=60, burst=10)
        bucket = _TokenBucket(rate=1.0, burst=10)
        middleware._buckets["5.6.7.8"] = bucket

        middleware._last_eviction = 0
        middleware._maybe_evict()

        assert "5.6.7.8" in middleware._buckets

    def test_eviction_respects_interval(self):
        """Eviction should not run if called within the interval."""
        middleware = RateLimitMiddleware(app=None, requests_per_minute=60, burst=10)
        bucket = _TokenBucket(rate=1.0, burst=10)
        bucket.last_refill = time.monotonic() - 700
        middleware._buckets["1.2.3.4"] = bucket

        # last_eviction is recent → should skip eviction
        middleware._last_eviction = time.monotonic()
        middleware._maybe_evict()

        # Bucket should still be present (eviction skipped)
        assert "1.2.3.4" in middleware._buckets
