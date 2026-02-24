"""Tavily web search adapter – implements WebSearchPort."""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

from tavily import AsyncTavilyClient
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


class TavilyAdapter:
    """Wraps the Tavily search API and satisfies ``WebSearchPort``."""

    def __init__(self, settings: Settings) -> None:
        if not settings.TAVILY_API_KEY:
            raise AdapterError(
                "TAVILY_API_KEY is not configured. Set it in .env or as an environment variable."
            )
        self._client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)

    async def search(self, query: str, max_results: int = 5) -> list[EvidenceChunk]:
        """Search the web via Tavily and return web_fallback chunks."""
        try:
            return await self._search_with_retry(query, max_results)
        except Exception as exc:
            logger.exception("Tavily search failed after retries")
            raise AdapterError(f"Tavily search error: {exc}") from exc

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
    async def _search_with_retry(self, query: str, max_results: int) -> list[EvidenceChunk]:
        """Invoke Tavily search with exponential backoff retry."""
        response = await self._client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        chunks: list[EvidenceChunk] = []
        for result in response.get("results", []):
            url = result.get("url", "")
            domain = urlparse(url).netloc or "unknown"
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
            source_id = f"WEB_{domain}_{url_hash}"

            chunk = EvidenceChunk(
                content=result.get("content", ""),
                source_id=source_id,
                source_type="web_fallback",
                metadata={
                    "url": url,
                    "title": result.get("title", ""),
                    "source_url": url,
                },
                relevance_score=result.get("score", 0.5),
            )
            chunks.append(chunk)
        return chunks
