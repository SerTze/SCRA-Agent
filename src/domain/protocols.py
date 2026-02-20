"""Domain protocols (interfaces) – depend on nothing outside the domain layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from src.domain.models import EvidenceChunk


# ---------------------------------------------------------------------------
# LLM Port
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMPort(Protocol):
    """Abstraction over any LLM backend (Groq, OpenAI, etc.)."""

    async def generate(self, prompt: str, *, system_prompt: str = "") -> str:
        """Return the raw text completion for the given prompt."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        schema: type[BaseModel],
    ) -> BaseModel:
        """Return a validated Pydantic object using provider-enforced structured output."""
        ...


# ---------------------------------------------------------------------------
# Retriever Port
# ---------------------------------------------------------------------------
@runtime_checkable
class RetrieverPort(Protocol):
    """Abstraction over a vector-store retriever."""

    async def retrieve(self, query: str, top_k: int = 25) -> list[EvidenceChunk]:
        """Return the top-k evidence chunks for a query."""
        ...


# ---------------------------------------------------------------------------
# Reranker Port
# ---------------------------------------------------------------------------
@runtime_checkable
class RerankerPort(Protocol):
    """Abstraction over a reranking service (Cohere, etc.)."""

    async def rerank(
        self,
        query: str,
        documents: list[EvidenceChunk],
        top_n: int = 5,
    ) -> list[EvidenceChunk]:
        """Re-score and truncate the document list."""
        ...


# ---------------------------------------------------------------------------
# Web Search Port  (Tavily fallback)
# ---------------------------------------------------------------------------
@runtime_checkable
class WebSearchPort(Protocol):
    """Abstraction over an external web search provider."""

    async def search(self, query: str, max_results: int = 5) -> list[EvidenceChunk]:
        """Return web-fallback evidence chunks."""
        ...


# ---------------------------------------------------------------------------
# Vector Store Port  (ingestion side)
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStorePort(Protocol):
    """Abstraction for adding documents to the vector store."""

    async def add_documents(self, chunks: list[EvidenceChunk]) -> int:
        """Persist chunks and return count of documents added."""
        ...

    async def health_check(self) -> bool:
        """Return True if the store is reachable."""
        ...
