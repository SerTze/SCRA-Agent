"""Tests for ChunkingStrategy, _chunk_text, and _split_by_paragraph."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.config.settings import Settings
from src.domain.models import ChunkingStrategy
from src.infrastructure.ingestion import IngestionPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    strategy: ChunkingStrategy | None = None,
) -> IngestionPipeline:
    """Create a pipeline with test settings and the given strategy."""
    settings = Settings(
        GROQ_API_KEY="test",
        COHERE_API_KEY="test",
        TAVILY_API_KEY="test",
        CHROMA_PERSIST_DIR="./test_chroma_data",
        LOG_FILE="",
        LANGFUSE_ENABLED=False,
    )
    return IngestionPipeline(settings, chunking_strategy=strategy)


# ---------------------------------------------------------------------------
# ChunkingStrategy model tests
# ---------------------------------------------------------------------------


class TestChunkingStrategy:
    """Validate our Pydantic model for chunking config."""

    def test_defaults(self) -> None:
        cs = ChunkingStrategy()
        assert cs.name == "fixed_2000"
        assert cs.max_chars == 2000
        assert cs.overlap_chars == 200
        assert cs.split_mode == "fixed"
        assert cs.prepend_metadata is True

    def test_custom_values(self) -> None:
        cs = ChunkingStrategy(
            name="small_para",
            max_chars=800,
            overlap_chars=100,
            split_mode="paragraph",
            prepend_metadata=True,
        )
        assert cs.name == "small_para"
        assert cs.split_mode == "paragraph"
        assert cs.prepend_metadata is True

    def test_invalid_split_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChunkingStrategy(split_mode="sentence")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _chunk_text dispatcher tests
# ---------------------------------------------------------------------------


class TestChunkTextDispatcher:
    """Verify _chunk_text routes to the right splitter."""

    def test_fixed_mode_uses_split_text(self) -> None:
        pipeline = _make_pipeline(
            ChunkingStrategy(split_mode="fixed", max_chars=100, overlap_chars=20)
        )
        # Fixed splitter needs newlines between paragraphs
        text = "\n".join(f"Paragraph {i} content here." for i in range(20))
        chunks = pipeline._chunk_text(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 150  # some tolerance for overlap

    def test_paragraph_mode_uses_paragraph_splitter(self) -> None:
        pipeline = _make_pipeline(ChunkingStrategy(split_mode="paragraph", max_chars=500))
        text = "Introduction text.\n1. First requirement.\n2. Second requirement.\n3. Third requirement."
        chunks = pipeline._chunk_text(text)
        # With paragraph mode, the numbered items should be respected
        assert len(chunks) >= 1

    def test_prepend_metadata(self) -> None:
        pipeline = _make_pipeline(
            ChunkingStrategy(split_mode="fixed", max_chars=5000, prepend_metadata=True)
        )
        text = "Some content here."
        chunks = pipeline._chunk_text(text, section_heading="Article 5")
        assert chunks[0].startswith("Article 5\n")

    def test_prepend_metadata_disabled(self) -> None:
        pipeline = _make_pipeline(
            ChunkingStrategy(split_mode="fixed", max_chars=5000, prepend_metadata=False)
        )
        text = "Some content here."
        chunks = pipeline._chunk_text(text, section_heading="Article 5")
        assert not chunks[0].startswith("Article 5\n")

    def test_no_heading_no_prepend(self) -> None:
        pipeline = _make_pipeline(
            ChunkingStrategy(split_mode="fixed", max_chars=5000, prepend_metadata=True)
        )
        text = "Some content here."
        chunks = pipeline._chunk_text(text, section_heading="")
        # No heading → no prepend even if flag is set
        assert chunks[0] == "Some content here."


# ---------------------------------------------------------------------------
# _split_by_paragraph tests
# ---------------------------------------------------------------------------


class TestSplitByParagraph:
    """Tests for the paragraph-level splitter."""

    def test_numbered_paragraphs_split(self) -> None:
        text = (
            "1. Providers must ensure quality.\n"
            "2. Deployers must monitor performance.\n"
            "3. Importers must verify conformity."
        )
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=500)
        # Everything fits in one chunk – should be merged
        assert len(chunks) == 1
        assert "1. Providers" in chunks[0]
        assert "3. Importers" in chunks[0]

    def test_large_paragraphs_force_split(self) -> None:
        text = "1. " + "word " * 600 + "\n2. Short paragraph."
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=500)
        assert len(chunks) >= 2

    def test_lettered_subparagraphs(self) -> None:
        text = (
            "Article heading.\n"
            "(a) First condition applies.\n"
            "(b) Second condition applies.\n"
            "(c) Third condition applies."
        )
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=500)
        assert len(chunks) >= 1
        # intro text preserved
        assert "Article heading" in chunks[0]

    def test_roman_numeral_subparagraphs(self) -> None:
        text = "1. Main paragraph.\n(i) sub-item one.\n(ii) sub-item two.\n(iii) sub-item three."
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=500)
        assert len(chunks) >= 1

    def test_no_markers_falls_back_to_fixed(self) -> None:
        # Use newline-separated sentences so fixed splitter can split on \n
        text = "\n".join(f"This is sentence number {i} with no legal markers." for i in range(50))
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=200, overlap=20)
        # Should still produce chunks (via fixed fallback)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 250  # tolerance

    def test_small_segments_merged(self) -> None:
        text = "(a) Yes.\n(b) No.\n(c) Maybe.\n(d) Perhaps.\n(e) Certainly."
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=500)
        # All are tiny – should be merged into one chunk
        assert len(chunks) == 1

    def test_max_chars_respected(self) -> None:
        parts = [f"({chr(ord('a') + i)}) " + "x " * 100 for i in range(10)]
        text = "\n".join(parts)
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=400)
        for c in chunks:
            # Allow some slack for overlap / joining
            assert len(c) <= 500

    def test_empty_text(self) -> None:
        chunks = IngestionPipeline._split_by_paragraph("", max_chars=500)
        assert len(chunks) == 1

    def test_intro_before_first_marker(self) -> None:
        text = (
            "This regulation applies to all providers.\n1. First condition.\n2. Second condition."
        )
        chunks = IngestionPipeline._split_by_paragraph(text, max_chars=5000)
        assert len(chunks) == 1
        assert "This regulation" in chunks[0]
        assert "1. First" in chunks[0]


# ---------------------------------------------------------------------------
# Pipeline defaults (no strategy = default)
# ---------------------------------------------------------------------------


class TestPipelineStrategyDefaults:
    """Verify pipeline uses default strategy when none supplied."""

    def test_default_strategy(self) -> None:
        pipeline = _make_pipeline()
        assert pipeline._strategy.name == "fixed_2000"
        assert pipeline._strategy.split_mode == "fixed"
        assert pipeline._strategy.max_chars == 2000

    def test_custom_strategy_accepted(self) -> None:
        cs = ChunkingStrategy(name="test", max_chars=500, split_mode="paragraph")
        pipeline = _make_pipeline(cs)
        assert pipeline._strategy.name == "test"
        assert pipeline._strategy.split_mode == "paragraph"
