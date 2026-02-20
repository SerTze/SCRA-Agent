"""FastAPI application – thin HTTP layer over the LangGraph workflow."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

from src.application.cache import QueryCache
from src.application.workflow import build_workflow
from src.config.settings import Settings
from src.container import Container
from src.domain.models import ComplianceAnalysis, GraphState
from src.infrastructure.telemetry import flush_langfuse, get_langfuse_callback
from src.presentation.middleware import LatencyBudgetMiddleware
from src.presentation.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


_LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** This response is AI-generated and does not "
    "constitute legal advice. Consult a qualified legal professional "
    "for regulatory compliance decisions."
)


class QueryResponse(BaseModel):
    answer: str
    grounding_score: str
    compliance: ComplianceAnalysis | None = None
    sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    fallback_used: bool = False


class HealthResponse(BaseModel):
    status: str
    chroma_ok: bool
    doc_count: int


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int = 0
    task_id: str | None = None


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Manage app startup/shutdown lifecycle."""
    yield
    # Shutdown: close async HTTP clients, flush Langfuse events, and close
    # the tracer provider so BatchSpanProcessor doesn't write to a closed
    # stdout during exit.
    container: Container | None = getattr(app.state, "container", None)
    if container is not None:
        for adapter_name in ("cohere_adapter", "tavily_adapter"):
            adapter = vars(container).get(adapter_name)
            if adapter is not None and hasattr(adapter, "aclose"):
                try:
                    await adapter.aclose()
                except Exception:
                    logger.debug("Failed to close %s (non-fatal)", adapter_name, exc_info=True)

    flush_langfuse()
    try:
        from opentelemetry import trace as _trace

        provider = _trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        logger.debug("Tracer provider shutdown failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application with DI container."""
    settings = settings or Settings()

    app = FastAPI(
        title="SCRA – Self-Correcting Regulatory Agent",
        version="4.0.0",
        description="Compliance-aware regulatory Q&A agent with self-auditing loop",
        lifespan=_lifespan,
    )

    # -- Latency budget middleware --
    app.add_middleware(
        LatencyBudgetMiddleware,
        budget_seconds=settings.LATENCY_BUDGET_SECONDS,
    )

    # -- Rate limiting (applied after latency budget in middleware stack) --
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_RPM,
        burst=settings.RATE_LIMIT_BURST,
    )

    # -- DI Container --
    container = Container(settings)
    app.state.container = container

    # -- Ensure telemetry is configured before instrumentation --
    _ = container.tracer

    # -- Langfuse callback for LLM tracing (None when disabled) --
    langfuse_cb = get_langfuse_callback(settings)

    # -- Build compiled LangGraph workflow --
    compiled_workflow = build_workflow(
        retriever_service=container.retriever_service,
        generation_service=container.generation_service,
        grading_service=container.grading_service,
        citation_service=container.citation_service,
    )

    # -- OpenTelemetry --
    FastAPIInstrumentor.instrument_app(app)

    # -- Query response cache --
    query_cache = QueryCache(
        max_size=settings.CACHE_MAX_SIZE,
        ttl_seconds=settings.CACHE_TTL_SECONDS,
        similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
    )

    # -- Ingestion state tracking (bounded: entries expire after 1 hour) --
    _ingest_status: dict[str, dict] = {}
    _INGEST_STATUS_TTL = 3600.0  # 1 hour
    _INGEST_STATUS_MAX = 1000    # hard cap on tracked tasks

    def _cleanup_ingest_status() -> None:
        """Remove completed/failed entries older than TTL to prevent memory leaks."""
        now = time.time()
        stale = [
            tid for tid, info in _ingest_status.items()
            if info.get("status") != "running"
            and now - info.get("_created", 0) > _INGEST_STATUS_TTL
        ]
        for tid in stale:
            del _ingest_status[tid]
        # Hard cap: if still over limit, drop oldest non-running entries
        if len(_ingest_status) > _INGEST_STATUS_MAX:
            non_running = [
                (tid, info.get("_created", 0))
                for tid, info in _ingest_status.items()
                if info.get("status") != "running"
            ]
            non_running.sort(key=lambda x: x[1])
            for tid, _ in non_running[: len(_ingest_status) - _INGEST_STATUS_MAX]:
                del _ingest_status[tid]

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest) -> QueryResponse:
        """Run the full self-correcting agent loop."""
        # Check cache first
        cached = await query_cache.get(req.question)
        if cached is not None:
            logger.info("Cache hit for question (returning cached result)")
            return QueryResponse(**cached)

        start = time.perf_counter()

        initial_state = GraphState(
            question=req.question,
            max_retries=settings.MAX_RETRIES,
        )
        # Each loop cycle: refine + grade (2 nodes) vs initial retrieve +
        # generate + grade (3 nodes). With max_retries=2 + web_fallback the
        # graph hits ~15 node executions max, so set a safe limit.
        try:
            invoke_config: dict = {"recursion_limit": 40}
            if langfuse_cb is not None:
                invoke_config["callbacks"] = [langfuse_cb]
            result = await compiled_workflow.ainvoke(
                initial_state,
                config=invoke_config,
            )
        except Exception as exc:
            logger.exception("Workflow execution failed")
            raise HTTPException(
                status_code=500,
                detail="Query processing failed. Check server logs for details.",
            ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000

        generation = result.get("generation", "") or ""
        cited_sources = result.get("cited_sources", []) or []

        compliance_data = result.get("compliance_analysis")
        compliance = None
        if compliance_data:
            if isinstance(compliance_data, dict):
                compliance = ComplianceAnalysis(**compliance_data)
            else:
                compliance = compliance_data

        response = QueryResponse(
            answer=generation + _LEGAL_DISCLAIMER,
            grounding_score=result.get("grounding_score", "unknown"),
            compliance=compliance,
            sources=sorted(set(cited_sources)),
            latency_ms=round(elapsed_ms, 1),
            fallback_used=result.get("fallback_active", False),
        )

        # Cache successful responses
        await query_cache.put(req.question, response.model_dump())

        return response

    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        """Health check – verifies ChromaDB connectivity."""
        chroma_ok = await container.chroma_adapter.health_check()
        doc_count = await container.chroma_adapter.collection_count()
        return HealthResponse(
            status="ok" if chroma_ok else "degraded",
            chroma_ok=chroma_ok,
            doc_count=doc_count,
        )

    @app.post("/ingest", response_model=IngestResponse, status_code=202)
    async def ingest_endpoint(background_tasks: BackgroundTasks) -> IngestResponse:
        """Trigger ingestion of the EU AI Act from EUR-Lex.

        Runs as a background task so the endpoint returns immediately
        with a ``task_id`` for status polling.
        """
        _cleanup_ingest_status()
        task_id = str(uuid.uuid4())
        _ingest_status[task_id] = {
            "status": "running",
            "chunks_ingested": 0,
            "_created": time.time(),
        }

        async def _run_ingest() -> None:
            try:
                chunks = await container.ingestion_pipeline.run()
                count = await container.chroma_adapter.add_documents(chunks)
                _ingest_status[task_id] = {
                    "status": "completed",
                    "chunks_ingested": count,
                    "_created": _ingest_status[task_id]["_created"],
                }
                logger.info("Background ingestion %s completed: %d chunks", task_id, count)
            except Exception:
                logger.exception("Background ingestion %s failed", task_id)
                _ingest_status[task_id] = {
                    "status": "failed",
                    "chunks_ingested": 0,
                    "_created": _ingest_status[task_id]["_created"],
                }

        background_tasks.add_task(_run_ingest)
        return IngestResponse(status="accepted", task_id=task_id)

    @app.get("/ingest/{task_id}", response_model=IngestResponse)
    async def ingest_status_endpoint(task_id: str) -> IngestResponse:
        """Check the status of a background ingestion task."""
        status = _ingest_status.get(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown task_id")
        return IngestResponse(
            status=status["status"],
            chunks_ingested=status["chunks_ingested"],
            task_id=task_id,
        )

    @app.get("/stats")
    async def stats_endpoint() -> dict:
        """Return LLM usage statistics for cost monitoring."""
        return {
            "llm_usage": container.llm_adapter.usage_summary,
            "grading_usage": (
                container.grading_adapter.usage_summary
                if container.grading_adapter is not container.llm_adapter
                else "(same as llm_adapter)"
            ),
            "cache_size": query_cache.size,
        }

    return app
