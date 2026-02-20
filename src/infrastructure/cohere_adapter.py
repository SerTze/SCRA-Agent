"""Cohere reranker adapter – implements RerankerPort."""

from __future__ import annotations

import logging

import cohere
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings
from src.domain.exceptions import AdapterError
from src.domain.models import EvidenceChunk

logger = logging.getLogger(__name__)


class CohereAdapter:
    """Wraps the Cohere Rerank API and satisfies ``RerankerPort``."""

    def __init__(self, settings: Settings) -> None:
        if not settings.COHERE_API_KEY:
            raise AdapterError(
                "COHERE_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )
        self._settings = settings
        self._client = cohere.AsyncClientV2(
            api_key=settings.COHERE_API_KEY,
            timeout=30,
        )
        self._model = settings.COHERE_RERANK_MODEL

    async def rerank(
        self,
        query: str,
        documents: list[EvidenceChunk],
        top_n: int = 5,
    ) -> list[EvidenceChunk]:
        """Re-score documents using Cohere rerank and return top_n."""
        if not documents:
            return []
        try:
            return await self._rerank_with_retry(query, documents, top_n)
        except Exception as exc:
            logger.exception("Cohere rerank failed after retries")
            raise AdapterError(f"Cohere rerank error: {exc}") from exc

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if hasattr(self._client, "close"):
            await self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _rerank_with_retry(
        self,
        query: str,
        documents: list[EvidenceChunk],
        top_n: int,
    ) -> list[EvidenceChunk]:
        """Invoke Cohere rerank with exponential backoff retry."""
        doc_texts = [chunk.content for chunk in documents]
        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(documents)),
        )
        reranked: list[EvidenceChunk] = []
        for result in response.results:
            chunk = documents[result.index].model_copy(
                update={"relevance_score": result.relevance_score}
            )
            reranked.append(chunk)
        return reranked
