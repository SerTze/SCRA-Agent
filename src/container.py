"""Dependency injection container – wires up all layers.

No global singletons. The Container is instantiated once at startup and
passed to the presentation layer. Adapters are never created inside nodes.

Uses ``cached_property`` for lazy initialization – adapters and services
are only instantiated when first accessed, enabling graceful degradation
when not all API keys are configured.
"""

from __future__ import annotations

from functools import cached_property

from src.application.services.citation_service import CitationService
from src.application.services.generation_service import GenerationService
from src.application.services.grading_service import GradingService
from src.application.services.retriever_service import RetrieverService
from src.config.settings import Settings
from src.domain.models import RetrievalSettings
from src.infrastructure.chroma_adapter import ChromaAdapter
from src.infrastructure.cohere_adapter import CohereAdapter
from src.infrastructure.groq_adapter import GroqAdapter
from src.infrastructure.ingestion import IngestionPipeline
from src.infrastructure.openai_adapter import OpenAIAdapter
from src.infrastructure.tavily_adapter import TavilyAdapter
from src.infrastructure.telemetry import configure_logging, configure_telemetry


class Container:
    """Composition root – assembles the full dependency graph.

    All adapters and services use ``@cached_property`` for lazy, on-demand
    initialization.  Telemetry and logging are configured eagerly since
    they must be ready before any other component logs or traces.

    Usage::

        settings = Settings()
        container = Container(settings)
        # Access container.retriever_service, etc. – created on first use.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

        # Logging and telemetry must be configured eagerly (global state)
        configure_logging(self.settings)

    # ── Infrastructure adapters (lazy) ───────────────────────────────────

    @cached_property
    def llm_adapter(self) -> GroqAdapter | OpenAIAdapter:
        """Primary LLM adapter based on ``LLM_PROVIDER`` setting.

        Supports ``"groq"`` (default) and ``"openai"``.
        """
        provider = self.settings.LLM_PROVIDER.lower()
        if provider == "openai":
            return OpenAIAdapter(self.settings)
        return GroqAdapter(self.settings)

    @cached_property
    def grading_adapter(self) -> GroqAdapter | OpenAIAdapter:
        """LLM adapter for grading tasks.

        Uses ``GROQ_GRADING_MODEL`` if configured (Groq-only), otherwise
        falls back to the primary ``llm_adapter``.
        """
        if self.settings.LLM_PROVIDER.lower() == "groq" and self.settings.GROQ_GRADING_MODEL:
            grading_settings = self.settings.model_copy(
                update={"GROQ_MODEL": self.settings.GROQ_GRADING_MODEL}
            )
            return GroqAdapter(grading_settings)
        return self.llm_adapter

    @cached_property
    def cohere_adapter(self) -> CohereAdapter:
        """Cohere reranker adapter for document re-scoring."""
        return CohereAdapter(self.settings)

    @cached_property
    def chroma_adapter(self) -> ChromaAdapter:
        """ChromaDB adapter for vector storage and retrieval."""
        return ChromaAdapter(self.settings)

    @cached_property
    def tavily_adapter(self) -> TavilyAdapter:
        """Tavily web search adapter for fallback evidence."""
        return TavilyAdapter(self.settings)

    @cached_property
    def tracer(self):
        """OpenTelemetry tracer instance (global provider is set as side-effect)."""
        return configure_telemetry(self.settings)

    # ── Domain value objects ─────────────────────────────────────────────

    @cached_property
    def retrieval_settings(self) -> RetrievalSettings:
        """Retrieval hyper-parameters bridged from application Settings."""
        return RetrievalSettings(
            TOP_K_RETRIEVAL=self.settings.TOP_K_RETRIEVAL,
            TOP_K_FINAL=self.settings.TOP_K_FINAL,
            PRIMARY_SOURCE_BOOST=self.settings.PRIMARY_SOURCE_BOOST,
            TOP_K_SIBLINGS=self.settings.TOP_K_SIBLINGS,
        )

    # ── Application services (lazy) ──────────────────────────────────────

    @cached_property
    def retriever_service(self) -> RetrieverService:
        """Orchestrates retrieval, reranking, sibling expansion, and boosting."""
        return RetrieverService(
            retriever=self.chroma_adapter,
            reranker=self.cohere_adapter,
            web_search=self.tavily_adapter,
            settings=self.retrieval_settings,
            sibling_retriever=self.chroma_adapter,
        )

    @cached_property
    def citation_service(self) -> CitationService:
        """Validates bi-directional citation contracts."""
        return CitationService()

    @cached_property
    def grading_service(self) -> GradingService:
        """Grounding and compliance grading via LLM.

        When ``GROQ_GRADING_MODEL`` is set, a separate adapter is used for
        compliance grading to reduce the "LLM agrees with itself" bias.
        """
        if self.grading_adapter is not self.llm_adapter:
            return GradingService(
                llm=self.grading_adapter,
                compliance_llm=self.grading_adapter,
            )
        return GradingService(llm=self.grading_adapter)

    @cached_property
    def generation_service(self) -> GenerationService:
        """RAG prompt building and answer generation."""
        return GenerationService(llm=self.llm_adapter)

    # ── Ingestion ────────────────────────────────────────────────────────

    @cached_property
    def ingestion_pipeline(self) -> IngestionPipeline:
        """EU AI Act download, parsing, and chunking pipeline."""
        return IngestionPipeline(self.settings)
