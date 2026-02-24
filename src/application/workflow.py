"""LangGraph workflow – the self-correcting regulatory agent loop.

Nodes are thin orchestrators that delegate to application services.
State is validated after every node via Pydantic.

Design principles:
    1. **Tiered acceptance** – ``grounded + compliant`` exits immediately;
       ``partial + compliant`` is also accepted (summarisation naturally
       paraphrases).  Only ``hallucinated`` or ``non-compliant`` results
       trigger refinement.
    2. **True self-correction** – the ``refine`` node feeds grading
       feedback (grounding score, compliance flags, reasoning) back to
       the LLM together with the previous answer and evidence, asking
       it to fix the *identified problems*.  No blind question rewriting.
    3. **Best-answer tracking** – every graded answer is scored and the
       best one seen so far is kept.  On retry exhaustion the best answer
       is returned instead of a hard error.
    4. **Single retry budget** – ``loop_step`` is never reset.  Web
       fallback simply adds web evidence and continues counting.
    5. **Query rewriting on fallback** – the ``web_fallback`` node
       rewrites the original question for better retrieval before the
       second pass, addressing retrieval-quality failures that answer
       refinement alone cannot fix.
    6. **Graceful degradation** – the error node is only reached when
       *zero* usable answers were generated (no cited sources at all).

Flow:
    retrieve → generate → grade → [decide]
        ├─ (grounded|partial) + compliant → END
        ├─ no cited_sources at all → web_fallback (once) or best_answer
        ├─ stall detected (no quality improvement) → web_fallback
        ├─ evidence insufficient (grounding reasoning) → web_fallback
        ├─ hallucinated or non-compliant → refine → grade (loop)
        └─ budget exhausted → return best answer seen (or error)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.application.services.citation_service import CitationService
from src.application.services.generation_service import GenerationService
from src.application.services.grading_service import GradingService
from src.application.services.retriever_service import RetrieverService
from src.domain.exceptions import CitationValidationError
from src.domain.models import GraphState

logger = logging.getLogger(__name__)

# Type alias for async workflow node functions
_AsyncNodeFn = Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]


# ---------------------------------------------------------------------------
# Quality ranking helper
# ---------------------------------------------------------------------------
_GROUNDING_RANK = {"grounded": 3, "partial": 2, "hallucinated": 1, "unknown": 0}


def _quality_rank(grounding_score: str, is_compliant: bool) -> int:
    """Return a numeric quality rank (higher = better) for comparison."""
    base = _GROUNDING_RANK.get(grounding_score, 0)
    return base * 2 + (1 if is_compliant else 0)


# ---------------------------------------------------------------------------
# Node factory – each node is a closure over injected services
# ---------------------------------------------------------------------------
def _make_retrieve_node(
    retriever_service: RetrieverService,
) -> _AsyncNodeFn:
    """Create the 'retrieve' node."""

    async def retrieve(state: GraphState) -> dict[str, Any]:
        logger.info("[retrieve] query=%s fallback=%s", state.question, state.fallback_active)
        docs = await retriever_service.retrieve_and_rank(
            state.question,
            use_web_fallback=state.fallback_active,
        )
        return {"documents": docs}

    return retrieve


def _make_generate_node(
    generation_service: GenerationService,
) -> _AsyncNodeFn:
    """Create the 'generate' node."""

    async def generate(state: GraphState) -> dict[str, Any]:
        logger.info("[generate] docs=%d", len(state.documents))
        result = await generation_service.generate_answer(state.question, state.documents)
        logger.info("[generate] cited_sources=%d", len(result.cited_sources))
        return {
            "generation": result.answer,
            "cited_sources": result.cited_sources,
        }

    return generate


def _make_grade_node(
    grading_service: GradingService,
    citation_service: CitationService,
) -> _AsyncNodeFn:
    """Create a combined 'grade' node.

    Runs grounding and compliance grading concurrently, then validates
    citations.  After grading, updates best-answer tracking if this
    attempt is better than any previous one.
    """

    async def grade(state: GraphState) -> dict[str, Any]:
        generation = state.generation or ""

        # Run grounding and compliance grading concurrently
        grounding_coro = grading_service.grade_grounding(generation, state.documents)
        compliance_coro = grading_service.grade_compliance(generation, state.documents)
        grounding_result, compliance_analysis = await asyncio.gather(
            grounding_coro, compliance_coro
        )
        grounding_score = grounding_result.score
        grounding_reasoning = grounding_result.reasoning

        # Validate structured citations (may downgrade grounding score)
        try:
            citation_service.validate_structured(state.cited_sources, state.documents)
        except CitationValidationError as exc:
            logger.warning("[grade] citation validation: %s", exc)
            if grounding_score == "grounded":
                grounding_score = "partial"

        logger.info(
            "[grade] grounding=%s compliant=%s",
            grounding_score,
            compliance_analysis.is_compliant,
        )

        # ── Build grading feedback for self-correction ──────────────
        feedback_parts = [f"Grounding: {grounding_score}"]
        if grounding_reasoning:
            feedback_parts.append(f"Grounding reasoning: {grounding_reasoning}")
        if compliance_analysis.risk_flags:
            feedback_parts.append(f"Compliance flags: {', '.join(compliance_analysis.risk_flags)}")
        if compliance_analysis.reasoning:
            feedback_parts.append(
                f"Compliance reasoning: {'; '.join(compliance_analysis.reasoning)}"
            )
        if not compliance_analysis.is_compliant:
            feedback_parts.append("Compliance: FAILED")

        updates: dict[str, Any] = {
            "grounding_score": grounding_score,
            "grounding_reasoning": grounding_reasoning,
            "evidence_insufficient": grounding_result.evidence_insufficient,
            "compliance_analysis": compliance_analysis,
            "grading_feedback": " | ".join(feedback_parts),
        }

        rank = _quality_rank(grounding_score, compliance_analysis.is_compliant)
        improved = rank > state.best_quality_rank and bool(state.cited_sources)
        updates["quality_improved"] = improved

        if improved:
            logger.info(
                "[grade] new best answer (rank %d > %d)",
                rank,
                state.best_quality_rank,
            )
            updates.update(
                {
                    "best_generation": generation,
                    "best_grounding_score": grounding_score,
                    "best_compliance_analysis": compliance_analysis,
                    "best_cited_sources": list(state.cited_sources),
                    "best_quality_rank": rank,
                }
            )
        elif state.loop_step > 0:
            logger.warning(
                "[grade] stall detected – rank %d did not beat best %d",
                rank,
                state.best_quality_rank,
            )

        return updates

    return grade


def _make_refine_node(
    generation_service: GenerationService,
) -> _AsyncNodeFn:
    """Create the 'refine' node – the true self-correction mechanism.

    Feeds grading feedback (grounding score, compliance flags, reasoning)
    back to the LLM together with the previous answer and the same
    evidence, asking it to fix the identified problems.  No re-retrieval
    needed — the correction targets the *answer*, not the *query*.
    """

    async def refine(state: GraphState) -> dict[str, Any]:
        previous_answer = state.generation or ""
        ca = state.compliance_analysis

        grounding_reasoning = state.grounding_reasoning
        compliance_status = "unknown"
        compliance_flags: list[str] = []
        compliance_reasoning: list[str] = []

        if ca is not None:
            compliance_status = "compliant" if ca.is_compliant else "non-compliant"
            compliance_flags = ca.risk_flags
            compliance_reasoning = ca.reasoning

        logger.info(
            "[refine] fixing: grounding=%s compliant=%s flags=%s (step %d)",
            state.grounding_score,
            compliance_status,
            compliance_flags,
            state.loop_step,
        )

        result = await generation_service.refine_answer(
            question=state.original_question,
            documents=state.documents,
            previous_answer=previous_answer,
            grounding_score=state.grounding_score,
            grounding_reasoning=grounding_reasoning,
            compliance_status=compliance_status,
            compliance_flags=compliance_flags,
            compliance_reasoning=compliance_reasoning,
        )

        logger.info(
            "[refine] refined answer cited_sources=%d",
            len(result.cited_sources),
        )
        return {
            "generation": result.answer,
            "cited_sources": result.cited_sources,
            "loop_step": state.loop_step + 1,
        }

    return refine


def _make_web_fallback_node(
    generation_service: GenerationService,
) -> _AsyncNodeFn:
    """Create the 'web_fallback' node – rewrite query + activate Tavily.

    Rewrites the original question for better retrieval before the
    second pass.  This addresses retrieval-quality failures that
    refinement alone cannot fix.

    Does NOT reset loop_step – the retry budget is global.
    """

    async def web_fallback(state: GraphState) -> dict[str, Any]:
        logger.info("[web_fallback] activating after %d loops", state.loop_step)

        rewritten = await generation_service.rewrite_question(
            state.original_question,
        )
        logger.info(
            "[web_fallback] rewritten query: %s → %s",
            state.original_question,
            rewritten,
        )

        return {
            "fallback_active": True,
            "question": rewritten,
            # loop_step is NOT reset – single global budget
        }

    return web_fallback


def _make_best_answer_node() -> _AsyncNodeFn:
    """Terminal node – return the best answer seen so far with a caveat."""

    async def best_answer(state: GraphState) -> dict[str, Any]:
        logger.warning(
            "[best_answer] returning best of %d attempts (rank=%d)",
            state.loop_step,
            state.best_quality_rank,
        )
        return {
            "generation": state.best_generation,
            "grounding_score": state.best_grounding_score,
            "compliance_analysis": state.best_compliance_analysis,
            "cited_sources": state.best_cited_sources,
        }

    return best_answer


def _make_error_node() -> _AsyncNodeFn:
    """Terminal error node – only reached when no usable answer exists."""

    async def error_node(state: GraphState) -> dict[str, Any]:
        logger.error("[error] no usable answer produced")
        return {
            "generation": (
                "I was unable to produce a verified answer. "
                "Please refine your question or consult the EU AI Act directly."
            ),
            "grounding_score": "unknown",
        }

    return error_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Evidence insufficiency detection
# ---------------------------------------------------------------------------

# Legacy fallback markers – narrow phrases that specifically indicate a
# retrieval gap rather than hallucination descriptions.  Only phrases where
# the *evidence itself* is described as missing, not phrases where *claims*
# are described as unsupported.
_EVIDENCE_GAP_MARKERS = (
    "evidence is insufficient",
    "insufficient evidence to answer",
    "insufficient evidence to verify",
    "evidence does not contain",
    "evidence does not cover",
    "not present in the evidence",
    "no relevant evidence",
)


def _is_evidence_insufficient(state: GraphState) -> bool:
    """Check whether the grounding result signals a retrieval evidence gap.

    Uses the structured ``evidence_insufficient`` flag from the
    GroundingResult (set by the grading LLM).  Falls back to keyword
    matching on the reasoning string ONLY when the structured flag was
    never populated (i.e. legacy / non-structured LLM path where the
    flag stays at its default ``False``).

    The legacy markers are intentionally narrow to avoid false positives
    where the grader describes hallucinated claims using phrases like
    "not found in the provided evidence" — that's a hallucination signal,
    not a retrieval gap.
    """
    # Structured field – authoritative when set
    if state.evidence_insufficient:
        logger.info("[evidence_insufficient] triggered via structured flag")
        return True

    # Legacy fallback – narrow markers that specifically indicate a
    # retrieval gap rather than hallucination descriptions.
    reasoning = state.grounding_reasoning
    if not reasoning:
        return False
    lower = reasoning.lower()
    matched = any(marker in lower for marker in _EVIDENCE_GAP_MARKERS)
    if matched:
        logger.info("[evidence_insufficient] triggered via legacy keyword match in reasoning")
    return matched


# ---------------------------------------------------------------------------
# Decision functions (edges)
# ---------------------------------------------------------------------------
def _decide_after_grounding(state: GraphState) -> str:
    """Route after grounding + compliance grading.

    Acceptance tiers (best → worst):
        1. grounded + compliant → accept immediately
        2. partial  + compliant → accept (summarisation paraphrases)
        3. stall detected (quality didn't improve) → web_fallback
        4. evidence insufficient (grounding reasoning) → web_fallback
        5. anything else with cited_sources → refine (if budget left)
        6. no cited_sources → web_fallback (once) or best_answer/error
        7. budget exhausted → return best_answer if one exists, else error
    """
    # ── No evidence cited at all → fallback or finish ────────────────
    if not state.cited_sources:
        logger.info("[decide] no cited_sources – fallback path")
        if state.fallback_active:
            # Already tried web – return best or error
            return "best_answer" if state.best_quality_rank >= 0 else "error"
        return "web_fallback"

    ca = state.compliance_analysis
    is_compliant = ca.is_compliant if ca is not None else False

    # ── Tier 1: grounded + compliant → done ──────────────────────────
    if state.grounding_score == "grounded" and is_compliant:
        logger.info("[decide] grounded + compliant → accept")
        return "end"

    # ── Tier 2: partial + compliant → accept (good enough) ───────────
    if state.grounding_score == "partial" and is_compliant:
        logger.info("[decide] partial + compliant → accept (good enough)")
        return "end"

    # ── Budget check ─────────────────────────────────────────────────
    if state.loop_step >= state.max_retries:
        # Try web evidence once if not yet attempted
        if not state.fallback_active:
            logger.info("[decide] budget exhausted – trying web fallback")
            return "web_fallback"
        # Already tried everything – return best we have
        logger.info(
            "[decide] budget fully exhausted – returning best answer (rank=%d)",
            state.best_quality_rank,
        )
        return "best_answer" if state.best_quality_rank >= 0 else "error"

    # ── Stall detection: refinement didn't improve quality ───────────
    if state.loop_step > 0 and not state.quality_improved:
        logger.warning(
            "[decide] stall detected (quality did not improve) → escalating to %s",
            "web_fallback" if not state.fallback_active else "best_answer",
        )
        if not state.fallback_active:
            return "web_fallback"
        return "best_answer" if state.best_quality_rank >= 0 else "error"

    # ── Evidence insufficiency: refining can't fix missing evidence ──
    if _is_evidence_insufficient(state):
        logger.info(
            "[decide] grounding reasoning indicates insufficient evidence → escalating to %s",
            "web_fallback" if not state.fallback_active else "refine (no alt)",
        )
        if not state.fallback_active:
            return "web_fallback"
        # Already on web evidence; refine is the only option left

    # ── Needs improvement → refine ─────────────────────────────────
    logger.info(
        "[decide] %s / compliant=%s → refine (step %d/%d)",
        state.grounding_score,
        is_compliant,
        state.loop_step,
        state.max_retries,
    )
    return "refine"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_workflow(
    retriever_service: RetrieverService,
    generation_service: GenerationService,
    grading_service: GradingService,
    citation_service: CitationService,
) -> CompiledStateGraph:
    """Assemble and compile the LangGraph state-machine.

    Returns the compiled graph ready for ``ainvoke``.
    """
    workflow = StateGraph(GraphState)

    # -- Register nodes --
    workflow.add_node("retrieve", _make_retrieve_node(retriever_service))
    workflow.add_node("generate", _make_generate_node(generation_service))
    workflow.add_node("grade", _make_grade_node(grading_service, citation_service))
    workflow.add_node("refine", _make_refine_node(generation_service))
    workflow.add_node("web_fallback", _make_web_fallback_node(generation_service))
    workflow.add_node("best_answer", _make_best_answer_node())
    workflow.add_node("error", _make_error_node())

    # -- Edges --
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")

    workflow.add_conditional_edges(
        "grade",
        _decide_after_grounding,
        {
            "end": END,
            "refine": "refine",
            "web_fallback": "web_fallback",
            "best_answer": "best_answer",
            "error": "error",
        },
    )

    # Refine goes directly back to grade (no re-retrieval needed)
    workflow.add_edge("refine", "grade")
    # Web fallback re-retrieves with augmented evidence
    workflow.add_edge("web_fallback", "retrieve")
    workflow.add_edge("best_answer", END)
    workflow.add_edge("error", END)

    return workflow.compile()
