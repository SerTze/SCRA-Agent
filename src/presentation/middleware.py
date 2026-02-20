"""Latency-budget middleware – pure ASGI implementation.

Avoids BaseHTTPMiddleware limitations with streaming responses and
background tasks.  Enforces real async timeouts via asyncio.wait_for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.domain.exceptions import LatencyBudgetExceeded

logger = logging.getLogger(__name__)


_EXEMPT_PATHS = {"/ingest", "/health"}


class LatencyBudgetMiddleware:
    """Cancel and reject requests that exceed the configured latency budget.

    Pure ASGI middleware – works correctly with streaming responses and
    background tasks unlike ``BaseHTTPMiddleware``.
    """

    def __init__(self, app: ASGIApp, budget_seconds: float = 10.0) -> None:
        self.app = app
        self._budget = budget_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Normalise path with trailing-slash tolerance
        path = scope.get("path", "").rstrip("/")
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        response_started = False

        async def send_with_latency(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-latency-ms", f"{elapsed_ms:.0f}".encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            # NOTE: asyncio.wait_for cancels the wrapped coroutine on timeout,
            # but any synchronous work running inside asyncio.to_thread (e.g.
            # ChromaDB queries, blocking HTTP calls) will continue executing in
            # its thread until completion.  This is an inherent limitation of
            # thread-based async bridging — the 504 is sent to the client but
            # backend resources are not freed until the thread finishes.
            await asyncio.wait_for(
                self.app(scope, receive, send_with_latency),
                timeout=self._budget,
            )
        except TimeoutError:
            elapsed = time.perf_counter() - start
            exc = LatencyBudgetExceeded(elapsed, self._budget)
            logger.warning(
                "Latency budget exceeded: %.2fs > %.2fs for %s",
                elapsed,
                self._budget,
                path,
            )
            if not response_started:
                body = json.dumps({"detail": str(exc)}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 504,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"x-latency-ms", f"{elapsed * 1000:.0f}".encode()),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
