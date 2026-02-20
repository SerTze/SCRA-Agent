"""In-memory token-bucket rate limiter – pure ASGI middleware.

Protects LLM-calling endpoints from excessive requests that could
burn through API credits.  Uses a per-client-IP token bucket with
configurable rate and burst size.

This is intentionally simple and in-memory.  For multi-process or
distributed deployments, replace with Redis-backed rate limiting.
"""

from __future__ import annotations

import json
import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Paths that are always exempt from rate limiting
_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

# Buckets idle for longer than this are evicted to prevent memory leaks
# from bot/scan traffic with many unique IPs.
_BUCKET_TTL_SECONDS = 600.0  # 10 minutes
_EVICTION_INTERVAL = 60.0    # run eviction at most once per minute


class _TokenBucket:
    """Simple token-bucket implementation."""

    __slots__ = ("rate", "burst", "tokens", "last_refill")

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate          # tokens per second
        self.burst = burst        # max tokens
        self.tokens = float(burst)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def idle_seconds(self) -> float:
        """Seconds since last refill (i.e. last request from this client)."""
        return time.monotonic() - self.last_refill


class RateLimitMiddleware:
    """Per-client-IP rate limiting via token bucket.

    Parameters
    ----------
    app : ASGIApp
        The wrapped ASGI application.
    requests_per_minute : float
        Sustained request rate per client.  Default 30 req/min.
    burst : int
        Maximum burst size (bucket capacity).  Default 10.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: float = 30.0,
        burst: int = 10,
    ) -> None:
        self.app = app
        self._rate = requests_per_minute / 60.0   # convert to per-second
        self._burst = burst
        self._buckets: dict[str, _TokenBucket] = {}
        self._last_eviction = time.monotonic()

    def _client_ip(self, scope: Scope) -> str:
        """Extract client IP from ASGI scope."""
        client = scope.get("client")
        if client:
            return client[0]
        # Fallback for proxied connections
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-forwarded-for":
                return header_value.decode().split(",")[0].strip()
        return "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "").rstrip("/")
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        client_ip = self._client_ip(scope)

        # Lazy-create bucket for new IPs
        bucket = self._buckets.get(client_ip)
        if bucket is None:
            bucket = _TokenBucket(self._rate, self._burst)
            self._buckets[client_ip] = bucket

        # Periodically evict stale buckets to bound memory usage
        self._maybe_evict()

        if bucket.consume():
            await self.app(scope, receive, send)
            return

        # Rate limited – send 429
        logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
        body = json.dumps({
            "detail": "Rate limit exceeded. Please slow down.",
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", b"2"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    def _maybe_evict(self) -> None:
        """Remove idle buckets to prevent unbounded memory growth."""
        now = time.monotonic()
        if now - self._last_eviction < _EVICTION_INTERVAL:
            return
        self._last_eviction = now
        stale = [
            ip for ip, bucket in self._buckets.items()
            if bucket.idle_seconds > _BUCKET_TTL_SECONDS
        ]
        for ip in stale:
            del self._buckets[ip]
        if stale:
            logger.debug("Evicted %d stale rate-limit buckets", len(stale))
