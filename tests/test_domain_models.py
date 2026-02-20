"""Unit tests for domain models – no external dependencies."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.domain.models import (
    CITATION_PATTERN,
    SOURCE_ID_PATTERN,
    ComplianceAnalysis,
    EvidenceChunk,
    GenerationResult,
    GraphState,
    RetrievalSettings,
)


# ---------------------------------------------------------------------------
# EvidenceChunk
# ---------------------------------------------------------------------------
class TestEvidenceChunk:
    """Test EvidenceChunk creation and source_id validation."""

    def test_valid_article_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_Art5_Chunk0"

    def test_valid_article_section_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Art5_Sec1a_Chunk2",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_Art5_Sec1a_Chunk2"

    def test_valid_recital_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Rec23_Chunk1",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_Rec23_Chunk1"

    def test_valid_annex_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_AnnexIII_Chunk0",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_AnnexIII_Chunk0"

    def test_valid_page_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Page12_Chunk3",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_Page12_Chunk3"

    def test_valid_file_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Fileeu-ai-act_Chunk0",
            source_type="primary_legal",
        )
        assert chunk.source_id == "EUAI_Fileeu-ai-act_Chunk0"

    def test_valid_web_chunk(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="WEB_example.com_a1b2c3d4",
            source_type="web_fallback",
        )
        assert chunk.source_id == "WEB_example.com_a1b2c3d4"

    def test_invalid_source_id_rejects(self):
        with pytest.raises(ValidationError):
            EvidenceChunk(
                content="test",
                source_id="INVALID_ID",
                source_type="primary_legal",
            )

    def test_invalid_source_type_rejects(self):
        with pytest.raises(ValidationError):
            EvidenceChunk(
                content="test",
                source_id="EUAI_Art5_Chunk0",
                source_type="unknown_type",  # type: ignore[arg-type]
            )

    def test_default_relevance_score(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
        )
        assert chunk.relevance_score == 0.0

    def test_metadata_defaults_to_empty(self):
        chunk = EvidenceChunk(
            content="test",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
        )
        assert chunk.metadata == {}


# ---------------------------------------------------------------------------
# ComplianceAnalysis
# ---------------------------------------------------------------------------
class TestComplianceAnalysis:
    def test_defaults(self):
        analysis = ComplianceAnalysis(is_compliant=True)
        assert analysis.risk_flags == []
        assert analysis.reasoning == []

    def test_with_flags(self):
        analysis = ComplianceAnalysis(
            is_compliant=False,
            risk_flags=["misleading_risk_category"],
            reasoning=["Incorrectly classifies system as minimal risk."],
        )
        assert not analysis.is_compliant
        assert len(analysis.risk_flags) == 1


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------
class TestGraphState:
    def test_original_question_auto_set(self):
        state = GraphState(question="What is Article 5?")
        assert state.original_question == "What is Article 5?"

    def test_defaults(self):
        state = GraphState(question="test")
        assert state.documents == []
        assert state.generation is None
        assert state.cited_sources == []
        assert state.grounding_score == "unknown"
        assert state.compliance_analysis is None
        assert state.loop_step == 0
        assert state.max_retries == 3
        assert state.fallback_active is False


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------
class TestGenerationResult:
    def test_defaults(self):
        result = GenerationResult(answer="Hello")
        assert result.answer == "Hello"
        assert result.cited_sources == []

    def test_with_sources(self):
        result = GenerationResult(
            answer="See [EUAI_Art5_Chunk0].",
            cited_sources=["EUAI_Art5_Chunk0"],
        )
        assert result.cited_sources == ["EUAI_Art5_Chunk0"]

    def test_empty_sources_signals_insufficient(self):
        """An empty cited_sources list signals insufficient evidence."""
        result = GenerationResult(
            answer="The evidence is insufficient.",
            cited_sources=[],
        )
        assert result.cited_sources == []


# ---------------------------------------------------------------------------
# Citation regex
# ---------------------------------------------------------------------------
class TestCitationRegex:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("[EUAI_Art5_Chunk2]", ["EUAI_Art5_Chunk2"]),
            ("[EUAI_Rec23_Chunk1]", ["EUAI_Rec23_Chunk1"]),
            ("[EUAI_Page12_Chunk3]", ["EUAI_Page12_Chunk3"]),
            ("[WEB_example.com_a1b2c3d4]", ["WEB_example.com_a1b2c3d4"]),
            ("[EUAI_Art5_Sec1a_Chunk2]", ["EUAI_Art5_Sec1a_Chunk2"]),
            ("[EUAI_AnnexIII_Chunk0]", ["EUAI_AnnexIII_Chunk0"]),
            (
                "See [EUAI_Art5_Chunk2] and [EUAI_Rec23_Chunk1].",
                ["EUAI_Art5_Chunk2", "EUAI_Rec23_Chunk1"],
            ),
            ("[INVALID]", []),
        ],
    )
    def test_citation_pattern(self, text: str, expected: list[str]):
        assert CITATION_PATTERN.findall(text) == expected

    @pytest.mark.parametrize(
        "source_id,valid",
        [
            ("EUAI_Art5_Chunk0", True),
            ("EUAI_Rec23_Chunk1", True),
            ("WEB_example.com_a1b2c3d4", True),
            ("INVALID", False),
            ("EUAI_Art_Chunk0", False),  # missing article number
        ],
    )
    def test_source_id_pattern(self, source_id: str, valid: bool):
        match = SOURCE_ID_PATTERN.match(source_id)
        assert bool(match) == valid


# ---------------------------------------------------------------------------
# RetrievalSettings
# ---------------------------------------------------------------------------
class TestRetrievalSettings:
    def test_defaults(self):
        s = RetrievalSettings()
        assert s.TOP_K_RETRIEVAL == 25
        assert s.TOP_K_FINAL == 5
        assert s.PRIMARY_SOURCE_BOOST == 1.2
