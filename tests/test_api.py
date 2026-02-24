"""Unit tests for the FastAPI endpoints – all external APIs mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from src.config.settings import Settings
from src.presentation.api import create_app


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        GROQ_API_KEY="test",
        COHERE_API_KEY="test",
        TAVILY_API_KEY="test",
        CHROMA_PERSIST_DIR="./test_chroma_api",
        CHROMA_COLLECTION_NAME="test_api_collection",
        LATENCY_BUDGET_SECONDS=30.0,  # generous for tests
    )


@pytest.fixture()
def client(test_settings: Settings) -> TestClient:
    app = create_app(test_settings)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "chroma_ok" in data
        assert "doc_count" in data


class TestQueryEndpoint:
    def test_empty_question_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client: TestClient):
        resp = client.post("/query", json={})
        assert resp.status_code == 422


class TestQueryEndpointSuccess:
    def test_successful_query_returns_grounded_response(self, test_settings: Settings):
        """Mock the workflow to verify the full /query response contract."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "generation": (
                "Article 5 prohibits AI systems that deploy subliminal "
                "techniques [EUAI_Art5_Chunk0].\n\n"
                "Sources:\n- [EUAI_Art5_Chunk0]"
            ),
            "cited_sources": ["EUAI_Art5_Chunk0"],
            "grounding_score": "grounded",
            "compliance_analysis": {
                "is_compliant": True,
                "risk_flags": [],
                "reasoning": ["Accurate representation."],
            },
            "fallback_active": False,
        }

        with patch("src.presentation.api.build_workflow", return_value=mock_workflow):
            app = create_app(test_settings)
            client = TestClient(app)
            resp = client.post(
                "/query",
                json={"question": "What AI practices are prohibited under Article 5?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["grounding_score"] == "grounded"
        assert data["compliance"]["is_compliant"] is True
        assert "EUAI_Art5_Chunk0" in data["sources"]
        assert data["latency_ms"] > 0
        assert data["fallback_used"] is False


# ------------------------------------------------------------------
# /stats endpoint
# ------------------------------------------------------------------
class TestStatsEndpoint:
    def test_stats_returns_usage_and_cache(self, test_settings: Settings):
        """The /stats response must include llm_usage, grading_usage, and cache_size."""
        with patch("src.presentation.api.build_workflow", return_value=AsyncMock()):
            app = create_app(test_settings)
            client = TestClient(app)

        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_usage" in data
        assert "grading_usage" in data
        assert "cache_size" in data
        # Initially, cache should be empty
        assert data["cache_size"] == 0

    def test_stats_llm_usage_shape(self, test_settings: Settings):
        """llm_usage should contain the token-tracking keys."""
        with patch("src.presentation.api.build_workflow", return_value=AsyncMock()):
            app = create_app(test_settings)
            client = TestClient(app)

        data = client.get("/stats").json()
        usage = data["llm_usage"]
        assert "total_calls" in usage
        assert "total_prompt_tokens" in usage
        assert "total_completion_tokens" in usage
        assert "total_tokens" in usage


# ------------------------------------------------------------------
# Cache integration via /query
# ------------------------------------------------------------------
class TestQueryCacheIntegration:
    _WORKFLOW_RESULT = {
        "generation": (
            "Article 5 prohibits AI systems [EUAI_Art5_Chunk0].\n\nSources:\n- [EUAI_Art5_Chunk0]"
        ),
        "cited_sources": ["EUAI_Art5_Chunk0"],
        "grounding_score": "grounded",
        "compliance_analysis": {
            "is_compliant": True,
            "risk_flags": [],
            "reasoning": ["OK"],
        },
        "fallback_active": False,
    }

    def test_second_identical_query_is_cache_hit(self, test_settings: Settings):
        """A repeated question should be served from cache (workflow called once)."""
        mock_wf = AsyncMock()
        mock_wf.ainvoke.return_value = self._WORKFLOW_RESULT

        with patch("src.presentation.api.build_workflow", return_value=mock_wf):
            app = create_app(test_settings)
            client = TestClient(app)

            q = {"question": "What AI practices are prohibited under Article 5?"}
            resp1 = client.post("/query", json=q)
            resp2 = client.post("/query", json=q)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["answer"] == resp2.json()["answer"]
        # Workflow should only have been invoked once
        assert mock_wf.ainvoke.call_count == 1

    def test_cache_size_increments_after_query(self, test_settings: Settings):
        """After a successful /query, /stats should show cache_size=1."""
        mock_wf = AsyncMock()
        mock_wf.ainvoke.return_value = self._WORKFLOW_RESULT

        with patch("src.presentation.api.build_workflow", return_value=mock_wf):
            app = create_app(test_settings)
            client = TestClient(app)

            client.post(
                "/query",
                json={"question": "What AI practices are prohibited?"},
            )
            stats = client.get("/stats").json()

        assert stats["cache_size"] == 1


# ------------------------------------------------------------------
# /ingest endpoints
# ------------------------------------------------------------------
class TestIngestEndpoint:
    def test_ingest_post_returns_202_with_task_id(self, test_settings: Settings):
        """POST /ingest should return 202 Accepted and a task_id immediately."""
        with patch("src.presentation.api.build_workflow", return_value=AsyncMock()):
            app = create_app(test_settings)
            client = TestClient(app)

        resp = client.post("/ingest")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        assert data["task_id"] is not None

    def test_ingest_status_unknown_task_returns_404(self, test_settings: Settings):
        """GET /ingest/{task_id} with an unknown ID should return 404."""
        with patch("src.presentation.api.build_workflow", return_value=AsyncMock()):
            app = create_app(test_settings)
            client = TestClient(app)

        resp = client.get("/ingest/does-not-exist-task-id")
        assert resp.status_code == 404

    def test_ingest_status_known_task_returns_status(self, test_settings: Settings):
        """After POST /ingest, the returned task_id should be poll-able."""
        with patch("src.presentation.api.build_workflow", return_value=AsyncMock()):
            app = create_app(test_settings)
            client = TestClient(app)

        ingest_resp = client.post("/ingest")
        task_id = ingest_resp.json()["task_id"]

        status_resp = client.get(f"/ingest/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] in ("running", "completed", "failed")
        assert data["task_id"] == task_id
