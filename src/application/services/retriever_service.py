"""RetrieverService – retrieval + reranking + sibling expansion + boosting."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.domain.models import EvidenceChunk, RetrievalSettings
from src.domain.protocols import RerankerPort, RetrieverPort, WebSearchPort

logger = logging.getLogger(__name__)

# Section-type sub-priority weights
_SECTION_PRIORITY: dict[str, int] = {
    "article": 3,
    "annex": 3,
    "recital": 2,
}
_DEFAULT_SECTION_WEIGHT = 1
_SUB_BOOST_FACTOR = 0.05  # Multiplier for section weight sub-boost


@runtime_checkable
class SiblingRetrieverPort(Protocol):
    """Minimal interface for sibling chunk expansion."""

    async def retrieve_siblings(self, source_ids: list[str], k: int = 2) -> list[EvidenceChunk]: ...


class RetrieverService:
    """Orchestrates retrieval, reranking, sibling expansion, boosting."""

    def __init__(
        self,
        retriever: RetrieverPort,
        reranker: RerankerPort,
        web_search: WebSearchPort,
        settings: RetrievalSettings | None = None,
        sibling_retriever: SiblingRetrieverPort | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._web_search = web_search
        self._settings = settings or RetrievalSettings()
        self._sibling_retriever = sibling_retriever

    async def retrieve_and_rank(
        self,
        query: str,
        *,
        use_web_fallback: bool = False,
    ) -> list[EvidenceChunk]:
        """Full retrieval pipeline: retrieve → rerank → boost → return."""
        # Step 1 – vector retrieval
        raw_chunks = await self._retriever.retrieve(query, top_k=self._settings.TOP_K_RETRIEVAL)
        logger.info("Retrieved %d raw chunks from vector store", len(raw_chunks))

        # Step 2 – web fallback (only when explicitly requested)
        if use_web_fallback:
            web_chunks = await self._web_search.search(query, max_results=5)
            raw_chunks.extend(web_chunks)
            logger.info("Added %d web fallback chunks", len(web_chunks))

        if not raw_chunks:
            return []

        # Step 3 – rerank
        reranked = await self._reranker.rerank(query, raw_chunks, top_n=self._settings.TOP_K_FINAL)

        # Step 4 – sibling chunk expansion
        expanded = await self._expand_siblings(reranked)

        # Step 5 – source-priority boost
        boosted = self._apply_source_boost(expanded)

        # Step 6 – sort by boosted score descending
        boosted.sort(key=lambda c: c.relevance_score, reverse=True)

        return boosted

    # ------------------------------------------------------------------
    # Sibling chunk expansion
    # ------------------------------------------------------------------
    async def _expand_siblings(self, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        """Fetch adjacent chunks for top reranked results.

        Adds surrounding context from the same article / section
        without duplicating chunks already in the result set.
        """
        k = self._settings.TOP_K_SIBLINGS
        if not self._sibling_retriever or k <= 0 or not chunks:
            return chunks

        existing_ids = {c.source_id for c in chunks}
        source_ids = [c.source_id for c in chunks]

        try:
            siblings = await self._sibling_retriever.retrieve_siblings(source_ids, k=k)
        except Exception:
            logger.warning("Sibling expansion failed – continuing without", exc_info=True)
            return chunks

        added = 0
        for sib in siblings:
            if sib.source_id not in existing_ids:
                existing_ids.add(sib.source_id)
                chunks.append(sib)
                added += 1

        if added:
            logger.info("Sibling expansion added %d chunks", added)
        return chunks

    def _apply_source_boost(self, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        """Boost primary_legal chunks by PRIMARY_SOURCE_BOOST."""
        result: list[EvidenceChunk] = []
        for chunk in chunks:
            score = chunk.relevance_score
            if chunk.source_type == "primary_legal":
                score *= self._settings.PRIMARY_SOURCE_BOOST
            # Sub-priority by section
            section = chunk.metadata.get("section_type", "")
            section_weight = _SECTION_PRIORITY.get(section, _DEFAULT_SECTION_WEIGHT)
            score *= 1.0 + (section_weight - 1) * _SUB_BOOST_FACTOR  # small sub-boost
            result.append(chunk.model_copy(update={"relevance_score": score}))
        return result
