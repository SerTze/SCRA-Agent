"""Unit tests for CitationService."""

from __future__ import annotations

import pytest
from src.application.services.citation_service import CitationService
from src.domain.exceptions import CitationValidationError
from src.domain.models import EvidenceChunk


@pytest.fixture()
def service() -> CitationService:
    return CitationService()


@pytest.fixture()
def docs() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            content="Article 5 content",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
        ),
        EvidenceChunk(
            content="Recital 23 content",
            source_id="EUAI_Rec23_Chunk0",
            source_type="primary_legal",
        ),
    ]


# ---------------------------------------------------------------------------
# Structured validation (preferred path)
# ---------------------------------------------------------------------------
class TestValidateStructured:
    def test_valid_sources_pass(self, docs: list[EvidenceChunk]):
        assert (
            CitationService.validate_structured(["EUAI_Art5_Chunk0", "EUAI_Rec23_Chunk0"], docs)
            is True
        )

    def test_empty_sources_pass(self, docs: list[EvidenceChunk]):
        """Empty cited_sources is not an error – caller decides fallback."""
        assert CitationService.validate_structured([], docs) is True

    def test_unknown_source_raises(self, docs: list[EvidenceChunk]):
        with pytest.raises(CitationValidationError):
            CitationService.validate_structured(["EUAI_Art5_Chunk0", "EUAI_Art99_Chunk0"], docs)

    def test_single_valid_source(self, docs: list[EvidenceChunk]):
        assert CitationService.validate_structured(["EUAI_Art5_Chunk0"], docs) is True


# ---------------------------------------------------------------------------
# Legacy regex extraction
# ---------------------------------------------------------------------------


class TestExtractInlineCitations:
    def test_extracts_two_citations(self, service: CitationService):
        text = "See [EUAI_Art5_Chunk0] and [EUAI_Rec23_Chunk0]."
        assert service.extract_inline_citations(text) == [
            "EUAI_Art5_Chunk0",
            "EUAI_Rec23_Chunk0",
        ]

    def test_no_citations(self, service: CitationService):
        assert service.extract_inline_citations("No citations here.") == []


class TestExtractSourceBlockIds:
    def test_extracts_from_source_block(self, service: CitationService):
        text = "Answer text.\n\nSources:\n- [EUAI_Art5_Chunk0]\n- [EUAI_Rec23_Chunk0]\n"
        ids = service.extract_source_block_ids(text)
        assert set(ids) == {"EUAI_Art5_Chunk0", "EUAI_Rec23_Chunk0"}

    def test_no_source_block(self, service: CitationService):
        assert service.extract_source_block_ids("Just text.") == []


class TestValidate:
    def test_valid_citations_pass(self, service: CitationService, docs: list[EvidenceChunk]):
        text = (
            "Article 5 prohibits [EUAI_Art5_Chunk0]. "
            "Recital 23 explains [EUAI_Rec23_Chunk0].\n\n"
            "Sources:\n"
            "- [EUAI_Art5_Chunk0]\n"
            "- [EUAI_Rec23_Chunk0]\n"
        )
        assert service.validate(text, docs) is True

    def test_missing_source_block_entry_raises(
        self, service: CitationService, docs: list[EvidenceChunk]
    ):
        text = "Article 5 prohibits [EUAI_Art5_Chunk0].\n\nSources:\n- [EUAI_Rec23_Chunk0]\n"
        with pytest.raises(CitationValidationError):
            service.validate(text, docs)

    def test_source_not_used_inline_is_nonfatal(
        self, service: CitationService, docs: list[EvidenceChunk]
    ):
        """Sources listed but not cited inline should warn, not raise."""
        text = (
            "Article 5 prohibits stuff.\n\nSources:\n- [EUAI_Art5_Chunk0]\n- [EUAI_Rec23_Chunk0]\n"
        )
        # Should NOT raise – extra sources in the block are non-fatal
        assert service.validate(text, docs) is True

    def test_unknown_document_raises(self, service: CitationService, docs: list[EvidenceChunk]):
        text = "See [EUAI_Art99_Chunk0].\n\nSources:\n- [EUAI_Art99_Chunk0]\n"
        with pytest.raises(CitationValidationError):
            service.validate(text, docs)
