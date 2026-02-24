"""Unit tests for the LangGraph workflow logic."""

from __future__ import annotations

import pytest
from src.application.workflow import (
    _decide_after_grounding,
    _is_evidence_insufficient,
    _quality_rank,
)
from src.domain.models import ComplianceAnalysis, GraphState


class TestQualityRank:
    """Test the quality ranking helper."""

    def test_grounded_compliant_is_highest(self):
        assert _quality_rank("grounded", True) == 7

    def test_grounded_non_compliant(self):
        assert _quality_rank("grounded", False) == 6

    def test_partial_compliant(self):
        assert _quality_rank("partial", True) == 5

    def test_hallucinated_non_compliant(self):
        assert _quality_rank("hallucinated", False) == 2

    def test_unknown(self):
        assert _quality_rank("unknown", False) == 0

    def test_ranking_order(self):
        ranks = [
            _quality_rank("grounded", True),
            _quality_rank("grounded", False),
            _quality_rank("partial", True),
            _quality_rank("partial", False),
            _quality_rank("hallucinated", True),
            _quality_rank("hallucinated", False),
            _quality_rank("unknown", True),
            _quality_rank("unknown", False),
        ]
        assert ranks == sorted(ranks, reverse=True)


class TestDecideAfterGrounding:
    """Test the conditional edge routing logic."""

    # ── Tier 1: grounded + compliant → accept ─────────────────────────

    def test_grounded_and_compliant_ends(self):
        state = GraphState(
            question="q",
            grounding_score="grounded",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=True),
        )
        assert _decide_after_grounding(state) == "end"

    # ── Tier 2: partial + compliant → accept (good enough) ────────────

    def test_partial_and_compliant_ends(self):
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=True),
        )
        assert _decide_after_grounding(state) == "end"

    # ── Non-compliant → refine (if budget left) ─────────────────────

    def test_grounded_but_not_compliant_refines(self):
        state = GraphState(
            question="q",
            grounding_score="grounded",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=0,
        )
        assert _decide_after_grounding(state) == "refine"

    def test_partial_not_compliant_refines(self):
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=0,
        )
        assert _decide_after_grounding(state) == "refine"

    # ── Hallucinated → refine ────────────────────────────────────────

    def test_hallucinated_refines(self):
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["EUAI_Art5_Chunk0"],
            loop_step=0,
        )
        assert _decide_after_grounding(state) == "refine"

    # ── Budget exhausted → web_fallback (once) ────────────────────────

    def test_max_retries_triggers_web_fallback(self):
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["EUAI_Art5_Chunk0"],
            loop_step=2,
            max_retries=2,
        )
        assert _decide_after_grounding(state) == "web_fallback"

    # ── Budget exhausted + fallback already used → best_answer ────────

    def test_max_retries_with_fallback_returns_best_answer(self):
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["EUAI_Art5_Chunk0"],
            loop_step=2,
            max_retries=2,
            fallback_active=True,
            best_quality_rank=5,  # has a saved best answer
            best_generation="some answer",
        )
        assert _decide_after_grounding(state) == "best_answer"

    # ── Budget exhausted + no best answer → error ─────────────────────

    def test_max_retries_with_no_best_goes_to_error(self):
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["EUAI_Art5_Chunk0"],
            loop_step=2,
            max_retries=2,
            fallback_active=True,
            best_quality_rank=-1,  # nothing saved
        )
        assert _decide_after_grounding(state) == "error"

    # ── No cited_sources → web_fallback ───────────────────────────────

    def test_empty_cited_sources_triggers_web_fallback(self):
        state = GraphState(
            question="q",
            grounding_score="grounded",
            cited_sources=[],
            compliance_analysis=ComplianceAnalysis(is_compliant=True),
        )
        assert _decide_after_grounding(state) == "web_fallback"

    # ── No cited_sources + fallback active + best exists → best_answer ─

    def test_empty_cited_sources_with_fallback_returns_best_answer(self):
        state = GraphState(
            question="q",
            grounding_score="grounded",
            cited_sources=[],
            fallback_active=True,
            best_quality_rank=3,
            best_generation="some answer",
        )
        assert _decide_after_grounding(state) == "best_answer"

    # ── No cited_sources + fallback active + no best → error ──────────

    def test_empty_cited_sources_with_fallback_no_best_goes_to_error(self):
        state = GraphState(
            question="q",
            grounding_score="grounded",
            cited_sources=[],
            fallback_active=True,
            best_quality_rank=-1,
        )
        assert _decide_after_grounding(state) == "error"


class TestStallDetection:
    """Test that stalled refinement escalates instead of retrying."""

    def test_stall_on_second_attempt_triggers_web_fallback(self):
        """First refine didn't help → skip to web_fallback."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=1,
            quality_improved=False,
            best_quality_rank=4,
        )
        assert _decide_after_grounding(state) == "web_fallback"

    def test_stall_with_fallback_active_returns_best_answer(self):
        """Stalled even on web evidence → return best."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=1,
            quality_improved=False,
            fallback_active=True,
            best_quality_rank=4,
            best_generation="prev best",
        )
        assert _decide_after_grounding(state) == "best_answer"

    def test_stall_with_fallback_no_best_goes_to_error(self):
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=1,
            quality_improved=False,
            fallback_active=True,
            best_quality_rank=-1,
        )
        assert _decide_after_grounding(state) == "error"

    def test_no_stall_on_first_grading(self):
        """loop_step=0 should always allow refine even if quality_improved=False."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=0,
            quality_improved=False,  # shouldn’t matter at step 0
        )
        assert _decide_after_grounding(state) == "refine"

    def test_improvement_allows_refine(self):
        """If quality DID improve, continue refining."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["EUAI_Art5_Chunk0"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            loop_step=1,
            quality_improved=True,
            best_quality_rank=4,
        )
        assert _decide_after_grounding(state) == "refine"


class TestEvidenceInsufficientRouting:
    """Test that evidence-gap detection escalates to web_fallback."""

    def test_insufficient_evidence_structured_flag_triggers_web_fallback(self):
        """Structured evidence_insufficient=True triggers web_fallback."""
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="Some claims are not grounded.",
            evidence_insufficient=True,
            loop_step=0,
            quality_improved=True,
        )
        assert _decide_after_grounding(state) == "web_fallback"

    def test_insufficient_evidence_legacy_reasoning_triggers_web_fallback(self):
        """Legacy keyword matching on reasoning still works."""
        state = GraphState(
            question="q",
            grounding_score="hallucinated",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="The evidence is insufficient to answer the question. The evidence does not contain the relevant information.",
            loop_step=0,
            quality_improved=True,
        )
        assert _decide_after_grounding(state) == "web_fallback"

    def test_no_evidence_phrase_triggers_web_fallback(self):
        """Legacy marker 'evidence does not contain' triggers fallback."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="The evidence does not contain information about this topic.",
            loop_step=0,
            quality_improved=True,
        )
        assert _decide_after_grounding(state) == "web_fallback"

    def test_hallucination_description_does_not_trigger_fallback(self):
        """Phrases describing hallucinated claims should NOT trigger fallback."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="Points 6-8 are not found in the provided evidence chunks.",
            loop_step=0,
            quality_improved=True,
        )
        assert _decide_after_grounding(state) == "refine"

    def test_evidence_insufficient_with_fallback_active_still_refines(self):
        """If web evidence already active, no alternative — refine anyway."""
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="The evidence is insufficient to answer the question.",
            loop_step=0,
            quality_improved=True,
            fallback_active=True,
        )
        assert _decide_after_grounding(state) == "refine"

    def test_normal_reasoning_does_not_trigger_fallback(self):
        state = GraphState(
            question="q",
            grounding_score="partial",
            cited_sources=["x"],
            compliance_analysis=ComplianceAnalysis(is_compliant=False),
            grounding_reasoning="The answer paraphrases the source but is mostly accurate.",
            loop_step=0,
            quality_improved=True,
        )
        assert _decide_after_grounding(state) == "refine"


class TestIsEvidenceInsufficient:
    """Unit tests for the evidence insufficiency detection helper."""

    def test_structured_flag_true(self):
        """Structured field takes priority."""
        state = GraphState(
            question="q",
            evidence_insufficient=True,
            grounding_reasoning="",
        )
        assert _is_evidence_insufficient(state) is True

    def test_structured_flag_false_no_reasoning(self):
        state = GraphState(
            question="q",
            evidence_insufficient=False,
            grounding_reasoning="",
        )
        assert _is_evidence_insufficient(state) is False

    def test_empty_reasoning_no_flag(self):
        state = GraphState(question="q")
        assert _is_evidence_insufficient(state) is False

    @pytest.mark.parametrize(
        "phrase",
        [
            "The evidence is insufficient to answer this question.",
            "There is insufficient evidence to verify the categorisation.",
            "The evidence does not contain information about penalties.",
            "The evidence does not cover this topic.",
            "This detail is not present in the evidence.",
            "There is no relevant evidence in the provided chunks.",
        ],
    )
    def test_legacy_detects_markers(self, phrase: str):
        """Legacy keyword matching still works when structured flag is False."""
        state = GraphState(
            question="q",
            evidence_insufficient=False,
            grounding_reasoning=phrase,
        )
        assert _is_evidence_insufficient(state) is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "Points 6, 7, and 8 are not found in the provided evidence chunks.",
            "Some claims are not supported by the evidence.",
            "The answer lacks supporting evidence for two claims.",
            "Claims cannot be verified from the sources.",
        ],
    )
    def test_hallucination_descriptions_do_not_trigger(self, phrase: str):
        """Phrases describing hallucinated claims should NOT trigger the legacy path."""
        state = GraphState(
            question="q",
            evidence_insufficient=False,
            grounding_reasoning=phrase,
        )
        assert _is_evidence_insufficient(state) is False

    def test_rejects_normal_reasoning(self):
        state = GraphState(
            question="q",
            evidence_insufficient=False,
            grounding_reasoning="The answer correctly paraphrases Article 5.",
        )
        assert _is_evidence_insufficient(state) is False
