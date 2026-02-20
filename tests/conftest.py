"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os

import pytest
from src.config.settings import Settings
from src.domain.models import ComplianceAnalysis, EvidenceChunk, GraphState

# ---------------------------------------------------------------------------
# Skip marker for live-API tests
# ---------------------------------------------------------------------------
live_test = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS", "0") != "1",
    reason="Live tests disabled (set RUN_LIVE_TESTS=1 to enable)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance with safe defaults for testing."""
    return Settings(
        GROQ_API_KEY="test-key",
        COHERE_API_KEY="test-key",
        TAVILY_API_KEY="test-key",
        CHROMA_PERSIST_DIR="./test_chroma_data",
        CHROMA_COLLECTION_NAME="test_collection",
        LATENCY_BUDGET_SECONDS=10.0,
        LOG_FILE="",  # disable file logging in tests
        LANGFUSE_ENABLED=False,
    )


@pytest.fixture()
def sample_chunks() -> list[EvidenceChunk]:
    """Two valid evidence chunks for testing."""
    return [
        EvidenceChunk(
            content="Article 5 prohibits AI systems that deploy subliminal techniques.",
            source_id="EUAI_Art5_Chunk0",
            source_type="primary_legal",
            metadata={
                "regulation": "EU_AI_ACT",
                "year": "2024",
                "section_type": "article",
                "section_number": "5",
                "source_url": "https://eur-lex.europa.eu/...",
            },
            relevance_score=0.95,
        ),
        EvidenceChunk(
            content="Recital 23 explains the scope of prohibited AI practices.",
            source_id="EUAI_Rec23_Chunk0",
            source_type="primary_legal",
            metadata={
                "regulation": "EU_AI_ACT",
                "year": "2024",
                "section_type": "recital",
                "section_number": "23",
                "source_url": "https://eur-lex.europa.eu/...",
            },
            relevance_score=0.80,
        ),
    ]


@pytest.fixture()
def sample_state(sample_chunks: list[EvidenceChunk]) -> GraphState:
    """A pre-populated GraphState for testing."""
    return GraphState(
        question="What AI practices are prohibited under Article 5?",
        documents=sample_chunks,
        generation=(
            "Article 5 prohibits AI systems that deploy subliminal techniques "
            "[EUAI_Art5_Chunk0]. Recital 23 provides further context [EUAI_Rec23_Chunk0].\n\n"
            "Sources:\n"
            "- [EUAI_Art5_Chunk0]\n"
            "- [EUAI_Rec23_Chunk0]"
        ),
        cited_sources=["EUAI_Art5_Chunk0", "EUAI_Rec23_Chunk0"],
        grounding_score="grounded",
        compliance_analysis=ComplianceAnalysis(
            is_compliant=True, risk_flags=[], reasoning=["Accurate representation."]
        ),
    )
