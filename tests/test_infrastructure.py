"""Unit tests for infrastructure adapters – all external APIs are mocked."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from src.domain.exceptions import AdapterError
from src.domain.models import EvidenceChunk
from src.infrastructure.base_llm_adapter import BaseLLMAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(source_id: str = "EUAI_Art5_Chunk0", content: str = "text") -> EvidenceChunk:
    return EvidenceChunk(
        content=content,
        source_id=source_id,
        source_type="primary_legal",
        metadata={"source_url": "http://example.com"},
    )


class _StubSchema(BaseModel):
    """Minimal Pydantic model for structured-output tests."""

    answer: str


class _FakeClient:
    """Minimal mock standing in for a LangChain ChatModel."""

    def __init__(self, reply_text: str = "Hello") -> None:
        self._reply = reply_text
        self.ainvoke = AsyncMock(return_value=self._response())

    def _response(self):
        return SimpleNamespace(
            content=self._reply,
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
            response_metadata={},
        )

    def with_structured_output(self, schema, *, include_raw=False):
        parsed = schema(answer=self._reply)
        raw = self._response()
        structured_mock = AsyncMock(
            return_value={"raw": raw, "parsed": parsed, "parsing_error": None}
        )
        return SimpleNamespace(ainvoke=structured_mock)


class _ConcreteAdapter(BaseLLMAdapter):
    """Concrete subclass of BaseLLMAdapter for testing."""

    def __init__(self, client=None, provider_name: str = "Test") -> None:
        self._client = client or _FakeClient()
        self._provider_name = provider_name
        super().__init__()


# ═══════════════════════════════════════════════════════════════════════════
# BaseLLMAdapter
# ═══════════════════════════════════════════════════════════════════════════
class TestBaseLLMAdapterGenerate:
    """Tests for BaseLLMAdapter.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        adapter = _ConcreteAdapter(client=_FakeClient("The answer"))
        result = await adapter.generate("What is 1+1?")
        assert result == "The answer"
        adapter._client.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_tracks_tokens(self) -> None:
        adapter = _ConcreteAdapter()
        await adapter.generate("question")
        usage = adapter.usage_summary
        assert usage["total_calls"] == 1
        assert usage["total_prompt_tokens"] == 10
        assert usage["total_completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_accumulates_across_calls(self) -> None:
        adapter = _ConcreteAdapter()
        await adapter.generate("q1")
        await adapter.generate("q2")
        assert adapter.usage_summary["total_calls"] == 2
        assert adapter.usage_summary["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_raises_adapter_error_on_failure(self) -> None:
        client = _FakeClient()
        client.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        adapter = _ConcreteAdapter(client=client)
        with pytest.raises(AdapterError, match="generation error"):
            await adapter.generate("boom")

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        adapter = _ConcreteAdapter()
        await adapter.generate("q", system_prompt="You are a lawyer")
        call_args = adapter._client.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system + human
        assert "lawyer" in call_args[0].content


class TestBaseLLMAdapterStructured:
    """Tests for BaseLLMAdapter.generate_structured()."""

    @pytest.mark.asyncio
    async def test_structured_returns_parsed_model(self) -> None:
        adapter = _ConcreteAdapter()
        result = await adapter.generate_structured("q", schema=_StubSchema)
        assert isinstance(result, _StubSchema)
        assert result.answer == "Hello"

    @pytest.mark.asyncio
    async def test_structured_caches_llm_wrapper(self) -> None:
        adapter = _ConcreteAdapter()
        await adapter.generate_structured("q1", schema=_StubSchema)
        await adapter.generate_structured("q2", schema=_StubSchema)
        # with_structured_output should be called only once (cached)
        assert _StubSchema in adapter._structured_cache

    @pytest.mark.asyncio
    async def test_structured_raises_adapter_error_on_failure(self) -> None:
        client = _FakeClient()
        broken_struct = AsyncMock(side_effect=RuntimeError("parse fail"))
        client.with_structured_output = MagicMock(
            return_value=SimpleNamespace(ainvoke=broken_struct)
        )
        adapter = _ConcreteAdapter(client=client)
        with pytest.raises(AdapterError, match="structured generation error"):
            await adapter.generate_structured("boom", schema=_StubSchema)


class TestUsageSummary:
    """Tests for token tracking edge cases."""

    @pytest.mark.asyncio
    async def test_usage_with_response_metadata_fallback(self) -> None:
        """Track tokens when usage_metadata is absent but response_metadata is present."""
        resp = SimpleNamespace(
            content="ok",
            response_metadata={"token_usage": {"prompt_tokens": 20, "completion_tokens": 8}},
        )
        adapter = _ConcreteAdapter()
        adapter._client.ainvoke = AsyncMock(return_value=resp)
        await adapter.generate("test")
        assert adapter.usage_summary["total_prompt_tokens"] == 20
        assert adapter.usage_summary["total_completion_tokens"] == 8

    @pytest.mark.asyncio
    async def test_usage_with_no_metadata(self) -> None:
        """Track tokens as zero when no metadata is present."""
        resp = SimpleNamespace(content="ok")
        adapter = _ConcreteAdapter()
        adapter._client.ainvoke = AsyncMock(return_value=resp)
        await adapter.generate("test")
        assert adapter.usage_summary["total_calls"] == 1
        assert adapter.usage_summary["total_tokens"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# ChromaAdapter
# ═══════════════════════════════════════════════════════════════════════════
class TestChromaAdapter:
    """Tests for ChromaAdapter with a mocked ChromaDB collection."""

    @pytest.fixture
    def _mock_settings(self):
        """Minimal Settings-like object for adapter construction."""
        s = MagicMock()
        s.CHROMA_PERSIST_DIR = "test_chroma_tmp"
        s.CHROMA_COLLECTION_NAME = "test_col"
        s.EMBEDDING_MODEL = "default"
        s.COHERE_API_KEY = ""
        return s

    @pytest.mark.asyncio
    async def test_retrieve_empty_collection(self, _mock_settings, tmp_path) -> None:
        _mock_settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")
        from src.infrastructure.chroma_adapter import ChromaAdapter

        adapter = ChromaAdapter(_mock_settings)
        # Empty collection → empty results
        results = await adapter.retrieve("What is Article 5?")
        assert results == []

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, _mock_settings, tmp_path) -> None:
        _mock_settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")
        from src.infrastructure.chroma_adapter import ChromaAdapter

        adapter = ChromaAdapter(_mock_settings)

        chunks = [
            _make_chunk("EUAI_Art5_Chunk0", "Prohibited AI practices include..."),
            _make_chunk("EUAI_Art6_Chunk0", "High-risk AI systems are..."),
        ]
        added = await adapter.add_documents(chunks)
        assert added == 2

        results = await adapter.retrieve("prohibited AI", top_k=2)
        assert len(results) >= 1
        assert all(isinstance(r, EvidenceChunk) for r in results)

    @pytest.mark.asyncio
    async def test_collection_count_is_async(self, _mock_settings, tmp_path) -> None:
        _mock_settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")
        from src.infrastructure.chroma_adapter import ChromaAdapter

        adapter = ChromaAdapter(_mock_settings)
        count = await adapter.collection_count()
        assert count == 0
        # Verify it's a coroutine (async)
        assert asyncio.iscoroutinefunction(adapter.collection_count)

    @pytest.mark.asyncio
    async def test_health_check(self, _mock_settings, tmp_path) -> None:
        _mock_settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")
        from src.infrastructure.chroma_adapter import ChromaAdapter

        adapter = ChromaAdapter(_mock_settings)
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_retrieve_siblings(self, _mock_settings, tmp_path) -> None:
        _mock_settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")
        from src.infrastructure.chroma_adapter import ChromaAdapter

        adapter = ChromaAdapter(_mock_settings)
        chunks = [
            _make_chunk("EUAI_Art5_Chunk0", "First chunk of Article 5"),
            _make_chunk("EUAI_Art5_Chunk1", "Second chunk of Article 5"),
            _make_chunk("EUAI_Art5_Chunk2", "Third chunk of Article 5"),
        ]
        await adapter.add_documents(chunks)

        siblings = await adapter.retrieve_siblings(["EUAI_Art5_Chunk1"], k=1)
        sibling_ids = {s.source_id for s in siblings}
        # Should get Chunk0 and Chunk2 (not Chunk1 itself)
        assert "EUAI_Art5_Chunk0" in sibling_ids or "EUAI_Art5_Chunk2" in sibling_ids


# ═══════════════════════════════════════════════════════════════════════════
# CohereAdapter
# ═══════════════════════════════════════════════════════════════════════════
class TestCohereAdapter:
    """Tests for CohereAdapter with mocked Cohere client."""

    @pytest.mark.asyncio
    async def test_rerank_returns_scored_chunks(self) -> None:
        from src.infrastructure.cohere_adapter import CohereAdapter

        settings = MagicMock()
        settings.COHERE_API_KEY = "fake-key"
        settings.COHERE_RERANK_MODEL = "rerank-v3.5"

        adapter = CohereAdapter(settings)

        # Mock the async Cohere client
        mock_result = SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.95),
                SimpleNamespace(index=0, relevance_score=0.80),
            ]
        )
        adapter._client.rerank = AsyncMock(return_value=mock_result)

        chunks = [
            _make_chunk("EUAI_Art5_Chunk0", "Article 5 on prohibited practices"),
            _make_chunk("EUAI_Art6_Chunk0", "Article 6 on high-risk systems"),
        ]
        result = await adapter.rerank("prohibited", chunks, top_n=2)
        assert len(result) == 2
        assert result[0].relevance_score == 0.95
        assert result[0].source_id == "EUAI_Art6_Chunk0"
        assert result[1].relevance_score == 0.80

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self) -> None:
        from src.infrastructure.cohere_adapter import CohereAdapter

        settings = MagicMock()
        settings.COHERE_API_KEY = "fake-key"
        settings.COHERE_RERANK_MODEL = "rerank-v3.5"
        adapter = CohereAdapter(settings)
        result = await adapter.rerank("query", [], top_n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_failure_raises_adapter_error(self) -> None:
        from src.infrastructure.cohere_adapter import CohereAdapter

        settings = MagicMock()
        settings.COHERE_API_KEY = "fake-key"
        settings.COHERE_RERANK_MODEL = "rerank-v3.5"
        adapter = CohereAdapter(settings)
        adapter._client.rerank = AsyncMock(side_effect=RuntimeError("API down"))
        with pytest.raises(AdapterError, match="Cohere rerank error"):
            await adapter.rerank("q", [_make_chunk()], top_n=1)

    @pytest.mark.asyncio
    async def test_aclose(self) -> None:
        from src.infrastructure.cohere_adapter import CohereAdapter

        settings = MagicMock()
        settings.COHERE_API_KEY = "fake-key"
        settings.COHERE_RERANK_MODEL = "rerank-v3.5"
        adapter = CohereAdapter(settings)
        adapter._client.close = AsyncMock()
        await adapter.aclose()
        adapter._client.close.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════════
# TavilyAdapter
# ═══════════════════════════════════════════════════════════════════════════
class TestTavilyAdapter:
    """Tests for TavilyAdapter with mocked Tavily client."""

    @pytest.mark.asyncio
    async def test_search_returns_web_chunks(self) -> None:
        from src.infrastructure.tavily_adapter import TavilyAdapter

        settings = MagicMock()
        settings.TAVILY_API_KEY = "fake-key"
        adapter = TavilyAdapter(settings)
        adapter._client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "url": "https://eur-lex.europa.eu/art5",
                        "title": "Article 5",
                        "content": "Prohibited practices under the EU AI Act",
                        "score": 0.9,
                    },
                ]
            }
        )

        results = await adapter.search("prohibited AI practices", max_results=1)
        assert len(results) == 1
        assert results[0].source_type == "web_fallback"
        assert results[0].source_id.startswith("WEB_")
        assert "eur-lex" in results[0].source_id

    @pytest.mark.asyncio
    async def test_search_failure_raises_adapter_error(self) -> None:
        from src.infrastructure.tavily_adapter import TavilyAdapter

        settings = MagicMock()
        settings.TAVILY_API_KEY = "fake-key"
        adapter = TavilyAdapter(settings)
        adapter._client.search = AsyncMock(side_effect=RuntimeError("Tavily down"))
        with pytest.raises(AdapterError, match="Tavily search error"):
            await adapter.search("q")

    @pytest.mark.asyncio
    async def test_aclose(self) -> None:
        from src.infrastructure.tavily_adapter import TavilyAdapter

        settings = MagicMock()
        settings.TAVILY_API_KEY = "fake-key"
        adapter = TavilyAdapter(settings)
        adapter._client.close = AsyncMock()
        await adapter.aclose()
        adapter._client.close.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════════
# LangfuseManager
# ═══════════════════════════════════════════════════════════════════════════
class TestLangfuseManager:
    """Tests for LangfuseManager (replacement for the module-level global)."""

    def test_returns_none_when_disabled(self) -> None:
        from src.infrastructure.telemetry import LangfuseManager

        mgr = LangfuseManager()
        settings = MagicMock()
        settings.LANGFUSE_ENABLED = False
        assert mgr.get_callback(settings) is None

    def test_returns_none_when_keys_missing(self) -> None:
        from src.infrastructure.telemetry import LangfuseManager

        mgr = LangfuseManager()
        settings = MagicMock()
        settings.LANGFUSE_ENABLED = True
        settings.LANGFUSE_SECRET_KEY = ""
        settings.LANGFUSE_PUBLIC_KEY = ""
        assert mgr.get_callback(settings) is None

    def test_flush_no_handler_is_noop(self) -> None:
        from src.infrastructure.telemetry import LangfuseManager

        mgr = LangfuseManager()
        mgr.flush()  # should not raise
