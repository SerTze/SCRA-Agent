"""Domain models for the Self-Correcting Regulatory Agent (SCRA)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Citation regex – single source of truth
# ---------------------------------------------------------------------------
_SOURCE_ID_INNER = (
    r"EUAI_Art\d+_Chunk\d+"
    r"|EUAI_Art\d+_Sec[A-Za-z0-9]+_Chunk\d+"
    r"|EUAI_Rec\d+_Chunk\d+"
    r"|EUAI_Annex[A-Za-z0-9]+_Chunk\d+"
    r"|EUAI_Page\d+_Chunk\d+"
    r"|EUAI_File[a-z0-9_-]+_Chunk\d+"
    r"|WEB_[A-Za-z0-9.-]+_[a-f0-9]{6,}"
)

CITATION_PATTERN = re.compile(rf"\[({_SOURCE_ID_INNER})\]")
SOURCE_ID_PATTERN = re.compile(rf"^({_SOURCE_ID_INNER})$")


# ---------------------------------------------------------------------------
# EvidenceChunk
# ---------------------------------------------------------------------------
class EvidenceChunk(BaseModel):
    """A single piece of evidence retrieved from the vector store."""

    content: str
    source_id: str
    source_type: Literal["primary_legal", "secondary_summary", "web_fallback"]
    metadata: dict[str, str] = Field(default_factory=dict)
    relevance_score: float = 0.0

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not SOURCE_ID_PATTERN.match(v):
            raise ValueError(
                f"source_id '{v}' does not match any allowed format. "
                "Expected EUAI_Art{{N}}_Chunk{{N}}, EUAI_Rec{{N}}_Chunk{{N}}, "
                "WEB_{{domain}}_{{hash}}, etc."
            )
        return v


# ---------------------------------------------------------------------------
# ComplianceAnalysis
# ---------------------------------------------------------------------------
class GroundingResult(BaseModel):
    """Structured result from the grounding grader."""

    score: Literal["grounded", "partial", "hallucinated", "unknown"] = "hallucinated"
    reasoning: str = ""
    evidence_insufficient: bool = False


class GenerationResult(BaseModel):
    """Structured result from the answer generation step.

    ``cited_sources`` must list source_ids actually referenced in the
    answer.  An empty list signals that the evidence was insufficient,
    which the workflow uses to trigger the web fallback path.
    """

    answer: str = Field(description="The generated regulatory compliance answer.")
    cited_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Source IDs referenced in the answer. "
            "Return an EMPTY list if the evidence is insufficient."
        ),
    )


class ComplianceAnalysis(BaseModel):
    """Result of the compliance grading step."""

    is_compliant: bool
    risk_flags: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphState  – the single state object flowing through LangGraph
# ---------------------------------------------------------------------------
class GraphState(BaseModel):
    """Mutable state object that flows through the LangGraph workflow."""

    question: str
    original_question: str = ""
    documents: list[EvidenceChunk] = Field(default_factory=list)
    generation: str | None = None
    cited_sources: list[str] = Field(default_factory=list)
    grounding_score: Literal["grounded", "partial", "hallucinated", "unknown"] = "unknown"
    compliance_analysis: ComplianceAnalysis | None = None
    loop_step: int = 0
    max_retries: int = 3
    fallback_active: bool = False

    # ── Grading feedback for self-correction ─────────────────────────
    grading_feedback: str = ""  # human-readable summary for refinement prompt
    grounding_reasoning: str = ""  # detailed reasoning from the grading LLM

    # ── Evidence insufficiency (structured flag from grounding grader) ──
    evidence_insufficient: bool = False

    # ── Stall detection ──────────────────────────────────────────────
    quality_improved: bool = True  # did the last grading improve over best?

    # ── Best-answer tracking (graceful degradation) ──────────────────
    best_generation: str | None = None
    best_grounding_score: str = "unknown"
    best_compliance_analysis: ComplianceAnalysis | None = None
    best_cited_sources: list[str] = Field(default_factory=list)
    best_quality_rank: int = -1  # higher is better; -1 = nothing saved

    def model_post_init(self, __context: object) -> None:
        if not self.original_question:
            self.original_question = self.question


# ---------------------------------------------------------------------------
# Retrieval settings value object
# ---------------------------------------------------------------------------
class RetrievalSettings(BaseModel):
    """Retrieval hyper-parameters – treated as a domain value object."""

    TOP_K_RETRIEVAL: int = 25
    TOP_K_FINAL: int = 5
    TOP_K_SIBLINGS: int = 2  # max sibling chunks to expand per retrieved chunk
    PRIMARY_SOURCE_BOOST: float = 1.2


# ---------------------------------------------------------------------------
# Chunking strategy – parameterises the ingestion pipeline
# ---------------------------------------------------------------------------
class ChunkingStrategy(BaseModel):
    """Configuration for how documents are split into chunks.

    Switch ``split_mode`` between strategies and compare retrieval
    quality using ``evals/eval_retrieval.py``.
    """

    name: str = "fixed_2000"
    max_chars: int = 2000
    overlap_chars: int = 200
    split_mode: Literal["fixed", "paragraph"] = "fixed"
    prepend_metadata: bool = True
