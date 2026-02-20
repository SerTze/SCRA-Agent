"""Unit tests for RetrieverService – external APIs are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.application.services.retriever_service import RetrieverService
from src.domain.models import EvidenceChunk, RetrievalSettings


@pytest.fixture()
def mock_retriever() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_reranker() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_web_search() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def service(
    mock_retriever: AsyncMock,
    mock_reranker: AsyncMock,
    mock_web_search: AsyncMock,
) -> RetrieverService:
    return RetrieverService(
        retriever=mock_retriever,
        reranker=mock_reranker,
        web_search=mock_web_search,
        settings=RetrievalSettings(),
    )


def _make_chunk(
    source_id: str = "EUAI_Art5_Chunk0",
    source_type: str = "primary_legal",
    score: float = 0.9,
    section_type: str = "article",
) -> EvidenceChunk:
    return EvidenceChunk(
        content="Test content",
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        metadata={"section_type": section_type},
        relevance_score=score,
    )


class TestRetrieveAndRank:
    @pytest.mark.asyncio
    async def test_basic_retrieval_flow(
        self,
        service: RetrieverService,
        mock_retriever: AsyncMock,
        mock_reranker: AsyncMock,
    ):
        chunks = [_make_chunk()]
        mock_retriever.retrieve.return_value = chunks
        mock_reranker.rerank.return_value = chunks

        result = await service.retrieve_and_rank("What is Article 5?")
        assert len(result) == 1
        mock_retriever.retrieve.assert_awaited_once()
        mock_reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_web_fallback_adds_chunks(
        self,
        service: RetrieverService,
        mock_retriever: AsyncMock,
        mock_reranker: AsyncMock,
        mock_web_search: AsyncMock,
    ):
        legal_chunk = _make_chunk()
        web_chunk = _make_chunk(
            source_id="WEB_example.com_aabbccdd",
            source_type="web_fallback",
            section_type="",
        )
        mock_retriever.retrieve.return_value = [legal_chunk]
        mock_web_search.search.return_value = [web_chunk]
        mock_reranker.rerank.return_value = [legal_chunk, web_chunk]

        result = await service.retrieve_and_rank("q", use_web_fallback=True)
        assert len(result) == 2
        mock_web_search.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_retrieval(
        self,
        service: RetrieverService,
        mock_retriever: AsyncMock,
    ):
        mock_retriever.retrieve.return_value = []
        result = await service.retrieve_and_rank("nothing")
        assert result == []

    @pytest.mark.asyncio
    async def test_primary_source_boost(
        self,
        service: RetrieverService,
        mock_retriever: AsyncMock,
        mock_reranker: AsyncMock,
    ):
        article = _make_chunk(score=0.8, section_type="article")
        web = _make_chunk(
            source_id="WEB_test.com_112233ee",
            source_type="web_fallback",
            score=0.85,
            section_type="",
        )
        mock_retriever.retrieve.return_value = [article, web]
        mock_reranker.rerank.return_value = [article, web]

        result = await service.retrieve_and_rank("q")
        # Article with boost (0.8 * 1.2 * 1.1 = 1.056) should rank above web (0.85)
        assert result[0].source_type == "primary_legal"
