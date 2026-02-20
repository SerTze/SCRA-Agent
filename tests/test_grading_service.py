"""Unit tests for GradingService – LLM calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.application.services.grading_service import GradingService
from src.domain.models import ComplianceAnalysis, EvidenceChunk, GroundingResult


@pytest.fixture()
def mock_llm() -> AsyncMock:
    """Mock LLM with both generate and generate_structured."""
    llm = AsyncMock()
    llm.generate_structured = AsyncMock()
    return llm


@pytest.fixture()
def mock_llm_legacy() -> AsyncMock:
    """Mock LLM without generate_structured (legacy path)."""
    llm = AsyncMock()
    # Remove generate_structured so GradingService falls back to legacy
    del llm.generate_structured
    return llm


@pytest.fixture()
def service(mock_llm: AsyncMock) -> GradingService:
    return GradingService(llm=mock_llm)


@pytest.fixture()
def service_legacy(mock_llm_legacy: AsyncMock) -> GradingService:
    return GradingService(llm=mock_llm_legacy)


@pytest.fixture()
def docs() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            content="Article 5 prohibits subliminal AI.",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
        )
    ]


class TestGradeGrounding:
    """Tests for structured output path."""

    @pytest.mark.asyncio
    async def test_grounded(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm.generate_structured.return_value = GroundingResult(
            score="grounded", reasoning="ok"
        )
        result = await service.grade_grounding("Answer text", docs)
        assert result.score == "grounded"
        assert result.reasoning == "ok"

    @pytest.mark.asyncio
    async def test_hallucinated(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm.generate_structured.return_value = GroundingResult(
            score="hallucinated", reasoning="no evidence"
        )
        result = await service.grade_grounding("Completely made up", docs)
        assert result.score == "hallucinated"

    @pytest.mark.asyncio
    async def test_structured_failure_falls_back_to_legacy(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        """When structured output raises, service falls back to generate + JSON."""
        mock_llm.generate_structured.side_effect = Exception("provider error")
        mock_llm.generate.return_value = '{"score": "partial", "reasoning": "fallback"}'
        result = await service.grade_grounding("Answer", docs)
        assert result.score == "partial"

    @pytest.mark.asyncio
    async def test_empty_docs_returns_unknown(self, service: GradingService):
        result = await service.grade_grounding("Answer", [])
        assert result.score == "unknown"


class TestGradeGroundingLegacy:
    """Tests for legacy (no generate_structured) path."""

    @pytest.mark.asyncio
    async def test_grounded_legacy(
        self, service_legacy: GradingService, mock_llm_legacy: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm_legacy.generate.return_value = '{"score": "grounded", "reasoning": "ok"}'
        result = await service_legacy.grade_grounding("Answer text", docs)
        assert result.score == "grounded"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_unknown(
        self, service_legacy: GradingService, mock_llm_legacy: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm_legacy.generate.return_value = "not json"
        result = await service_legacy.grade_grounding("Answer", docs)
        assert result.score == "unknown"


class TestGradeCompliance:
    """Tests for structured output path."""

    @pytest.mark.asyncio
    async def test_compliant(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm.generate_structured.return_value = ComplianceAnalysis(
            is_compliant=True, risk_flags=[], reasoning=["ok"]
        )
        result = await service.grade_compliance("Answer", docs)
        assert result.is_compliant is True
        assert result.risk_flags == []

    @pytest.mark.asyncio
    async def test_non_compliant(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm.generate_structured.return_value = ComplianceAnalysis(
            is_compliant=False,
            risk_flags=["misleading"],
            reasoning=["Incorrect risk classification"],
        )
        result = await service.grade_compliance("Bad answer", docs)
        assert result.is_compliant is False
        assert "misleading" in result.risk_flags

    @pytest.mark.asyncio
    async def test_structured_failure_falls_back_to_legacy(
        self, service: GradingService, mock_llm: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm.generate_structured.side_effect = Exception("provider error")
        mock_llm.generate.return_value = (
            '{"is_compliant": true, "risk_flags": [], "reasoning": ["fallback ok"]}'
        )
        result = await service.grade_compliance("Answer", docs)
        assert result.is_compliant is True

    @pytest.mark.asyncio
    async def test_empty_generation(self, service: GradingService):
        result = await service.grade_compliance("", [])
        assert result.is_compliant is False


class TestGradeComplianceLegacy:
    """Tests for legacy (no generate_structured) path."""

    @pytest.mark.asyncio
    async def test_compliant_legacy(
        self, service_legacy: GradingService, mock_llm_legacy: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm_legacy.generate.return_value = (
            '{"is_compliant": true, "risk_flags": [], "reasoning": ["ok"]}'
        )
        result = await service_legacy.grade_compliance("Answer", docs)
        assert result.is_compliant is True

    @pytest.mark.asyncio
    async def test_invalid_json_returns_parse_error(
        self, service_legacy: GradingService, mock_llm_legacy: AsyncMock, docs: list[EvidenceChunk]
    ):
        mock_llm_legacy.generate.return_value = "broken"
        result = await service_legacy.grade_compliance("Answer", docs)
        assert result.is_compliant is False
        assert "parse_error" in result.risk_flags


class TestParseJson:
    def test_plain_json(self):
        result = GradingService._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced(self):
        raw = '```json\n{"key": "value"}\n```'
        result = GradingService._parse_json(raw)
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        from src.domain.exceptions import LLMResponseParsingError

        with pytest.raises(LLMResponseParsingError):
            GradingService._parse_json("not json at all")
