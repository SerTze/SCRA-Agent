"""Integration tests – gated behind RUN_LIVE_TESTS=1.

These tests hit real APIs (Groq, Cohere, Tavily) and require valid keys.
"""

from __future__ import annotations

import pytest
from src.config.settings import Settings
from src.domain.models import EvidenceChunk
from src.infrastructure.cohere_adapter import CohereAdapter
from src.infrastructure.groq_adapter import GroqAdapter
from src.infrastructure.tavily_adapter import TavilyAdapter

from tests.conftest import live_test


@live_test
class TestLiveGroq:
    @pytest.mark.asyncio
    async def test_groq_generates_response(self):
        settings = Settings()
        adapter = GroqAdapter(settings)
        result = await adapter.generate("Say hello in one word.")
        assert len(result) > 0


@live_test
class TestLiveCohere:
    @pytest.mark.asyncio
    async def test_cohere_reranks(self):
        settings = Settings()
        adapter = CohereAdapter(settings)
        chunks = [
            EvidenceChunk(
                content="Article 5 prohibits subliminal AI techniques.",
                source_id="EUAI_Art5_Chunk0",
                source_type="primary_legal",
            ),
            EvidenceChunk(
                content="The weather is nice today.",
                source_id="EUAI_Art99_Chunk0",
                source_type="primary_legal",
            ),
        ]
        result = await adapter.rerank("prohibited AI practices", chunks, top_n=1)
        assert len(result) == 1
        assert result[0].source_id == "EUAI_Art5_Chunk0"


@live_test
class TestLiveTavily:
    @pytest.mark.asyncio
    async def test_tavily_search(self):
        settings = Settings()
        adapter = TavilyAdapter(settings)
        result = await adapter.search("EU AI Act Article 5", max_results=2)
        assert len(result) > 0
        assert result[0].source_type == "web_fallback"
