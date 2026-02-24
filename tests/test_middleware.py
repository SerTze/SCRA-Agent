"""Unit tests for LatencyBudgetMiddleware – pure ASGI, no real HTTP server."""

from __future__ import annotations

import asyncio
import json

from src.presentation.middleware import LatencyBudgetMiddleware


# ---------------------------------------------------------------------------
# ASGI test helpers
# ---------------------------------------------------------------------------
async def _run(
    app,
    path: str = "/query",
    scope_type: str = "http",
    budget: float = 5.0,
) -> list[dict]:
    """Drive middleware with a minimal ASGI scope and collect sent messages."""
    messages: list[dict] = []

    async def _send(msg: dict) -> None:
        messages.append(msg)

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": scope_type,
        "path": path,
        "method": "POST",
        "headers": [],
    }
    mw = LatencyBudgetMiddleware(app, budget_seconds=budget)
    await mw(scope, receive, _send)
    return messages


async def _ok_app(scope, receive, send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


# ═══════════════════════════════════════════════════════════════════════════
# Non-HTTP scope passthrough
# ═══════════════════════════════════════════════════════════════════════════
class TestNonHttpScope:
    async def test_lifespan_scope_passes_through_unchanged(self) -> None:
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        await _run(app, scope_type="lifespan")
        assert "lifespan" in reached

    async def test_websocket_scope_passes_through(self) -> None:
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        await _run(app, scope_type="websocket")
        assert "websocket" in reached


# ═══════════════════════════════════════════════════════════════════════════
# Exempt paths
# ═══════════════════════════════════════════════════════════════════════════
class TestExemptPaths:
    async def test_health_path_bypasses_budget(self) -> None:
        """The app is called directly without asyncio.wait_for for /health."""
        called_with = []

        async def app(scope, receive, send):
            called_with.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        await _run(app, path="/health")
        assert called_with == ["/health"]

    async def test_ingest_path_bypasses_budget(self) -> None:
        called_with = []

        async def app(scope, receive, send):
            called_with.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        await _run(app, path="/ingest")
        assert called_with == ["/ingest"]

    async def test_trailing_slash_exempt(self) -> None:
        """Paths with trailing slash should still be recognised as exempt."""
        called_with = []

        async def app(scope, receive, send):
            called_with.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        await _run(app, path="/health/")
        assert called_with == ["/health/"]


# ═══════════════════════════════════════════════════════════════════════════
# Normal request – latency header
# ═══════════════════════════════════════════════════════════════════════════
class TestLatencyHeader:
    async def test_x_latency_ms_header_appended(self) -> None:
        messages = await _run(_ok_app, path="/query")
        start = next(m for m in messages if m["type"] == "http.response.start")
        header_names = [k for k, _ in start["headers"]]
        assert b"x-latency-ms" in header_names

    async def test_latency_header_value_is_numeric(self) -> None:
        messages = await _run(_ok_app, path="/query")
        start = next(m for m in messages if m["type"] == "http.response.start")
        headers = {k: v for k, v in start["headers"]}
        ms = float(headers[b"x-latency-ms"])
        assert ms >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Timeout → 504
# ═══════════════════════════════════════════════════════════════════════════
class TestTimeout:
    async def test_slow_app_returns_504(self) -> None:
        async def slow_app(scope, receive, send):
            await asyncio.sleep(10)

        messages = await _run(slow_app, path="/query", budget=0.05)
        assert any(m.get("status") == 504 for m in messages)

    async def test_504_body_is_json_with_detail(self) -> None:
        async def slow_app(scope, receive, send):
            await asyncio.sleep(10)

        messages = await _run(slow_app, path="/query", budget=0.05)
        body_msg = next(m for m in messages if m.get("type") == "http.response.body")
        body = json.loads(body_msg["body"])
        assert "detail" in body

    async def test_504_includes_latency_header(self) -> None:
        async def slow_app(scope, receive, send):
            await asyncio.sleep(10)

        messages = await _run(slow_app, path="/query", budget=0.05)
        start = next(m for m in messages if m.get("status") == 504)
        header_names = [k for k, _ in start["headers"]]
        assert b"x-latency-ms" in header_names
