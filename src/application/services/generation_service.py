"""GenerationService – builds prompts and calls the LLM for answer generation."""

from __future__ import annotations

import logging
import re
import unicodedata

from src.application.prompts import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_USER_PROMPT,
)
from src.application.services.evidence_builder import (
    GENERATION_EVIDENCE_MAX_CHARS,
    build_evidence_block,
)
from src.domain.models import EvidenceChunk, GenerationResult
from src.domain.protocols import LLMPort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-injection defence (defense-in-depth)
#
# This is NOT a security boundary — regex blocklists are trivially bypassed
# with encoding tricks, synonyms, or multi-language attacks.  The real
# safeguards are: (1) the system prompt instructing the LLM to ignore
# override attempts, (2) grounding validation that rejects hallucinated
# output, and (3) structured-output constraints.  This filter catches the
# most common low-effort injection attempts and logs them for monitoring.
# ---------------------------------------------------------------------------
_INJECTION_RE = re.compile(
    r"ignore\s+(previous|above|all|prior|preceding)\s+(instructions|prompts?|context)"
    r"|disregard\s+(all\s+)?(previous|prior)"
    r"|you\s+are\s+now"
    r"|new\s+(instructions|role)\s*:"
    r"|system\s*prompt\s*:"
    r"|override\s*:"
    r"|forget\s+(everything|all|previous)"
    r"|act\s+as\s+(if|a|an)"
    r"|do\s+not\s+follow\s+(the|your|any)",
    re.IGNORECASE,
)


class GenerationService:
    """Builds the RAG prompt and invokes the LLM."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm
        self._supports_structured: bool = hasattr(llm, "generate_structured")

    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Remove known prompt injection patterns from user input.

        Applies NFKC normalization first to defeat Unicode homoglyph and
        encoding tricks (e.g. fullwidth Latin, Cyrillic look-alikes).
        """
        normalized = unicodedata.normalize("NFKC", text)
        cleaned = _INJECTION_RE.sub("[redacted]", normalized)
        if cleaned != normalized:
            logger.warning("Prompt injection pattern detected and redacted")
        return cleaned

    # Maximum chars of evidence to include in LLM prompts to stay
    # within token budgets (roughly 1 token ≈ 4 chars).
    _MAX_EVIDENCE_CHARS = GENERATION_EVIDENCE_MAX_CHARS

    async def generate_answer(
        self,
        question: str,
        documents: list[EvidenceChunk],
    ) -> GenerationResult:
        """Generate a cited answer from evidence chunks.

        Returns a ``GenerationResult`` with the answer text and a
        list of cited source IDs.  An empty ``cited_sources`` list
        signals insufficient evidence (triggers web fallback).
        """
        question = self._sanitize_input(question)
        evidence_block = self._build_evidence_block(documents)
        prompt = GENERATION_USER_PROMPT.format(
            question=question, evidence=evidence_block
        )

        # Prefer structured output when available
        if self._supports_structured:
            try:
                result = await self._llm.generate_structured(
                    prompt,
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    schema=GenerationResult,
                )
                if isinstance(result, GenerationResult):
                    return result
                # Defensive: in case the provider returns a dict
                return GenerationResult(**result)  # type: ignore[arg-type]
            except Exception:
                logger.warning(
                    "Structured generation failed – falling back to plain text",
                    exc_info=True,
                )

        # Fallback: plain text generation → wrap in GenerationResult
        raw = await self._llm.generate(
            prompt, system_prompt=GENERATION_SYSTEM_PROMPT
        )
        return GenerationResult(answer=raw, cited_sources=[])

    @classmethod
    def _build_evidence_block(
        cls,
        documents: list[EvidenceChunk],
        max_chars: int | None = None,
    ) -> str:
        """Delegate to the shared evidence builder."""
        return build_evidence_block(
            documents, max_chars=max_chars or cls._MAX_EVIDENCE_CHARS
        )

    async def rewrite_question(self, question: str) -> str:
        """Rewrite the question for better retrieval (query expansion)."""
        rewrite_prompt = (
            f"Rewrite the following question to improve retrieval from a legal "
            f"document corpus about the EU AI Act. Return ONLY the rewritten question.\n\n"
            f"Original: {question}"
        )
        return await self._llm.generate(rewrite_prompt)

    async def refine_answer(
        self,
        question: str,
        documents: list[EvidenceChunk],
        previous_answer: str,
        grounding_score: str,
        grounding_reasoning: str,
        compliance_status: str,
        compliance_flags: list[str],
        compliance_reasoning: list[str],
    ) -> GenerationResult:
        """Refine a previous answer using grading feedback.

        This is the core self-correction mechanism: the LLM sees its own
        previous output alongside specific feedback about what was wrong,
        and is asked to produce a corrected version.
        """
        evidence_block = self._build_evidence_block(documents)

        prompt = REFINEMENT_USER_PROMPT.format(
            question=question,
            evidence=evidence_block,
            previous_answer=previous_answer,
            grounding_score=grounding_score,
            grounding_reasoning=grounding_reasoning,
            compliance_status=compliance_status,
            compliance_flags=", ".join(compliance_flags) if compliance_flags else "none",
            compliance_reasoning="; ".join(compliance_reasoning) if compliance_reasoning else "none",
        )

        if self._supports_structured:
            try:
                result = await self._llm.generate_structured(
                    prompt,
                    system_prompt=REFINEMENT_SYSTEM_PROMPT,
                    schema=GenerationResult,
                )
                if isinstance(result, GenerationResult):
                    return result
                return GenerationResult(**result)  # type: ignore[arg-type]
            except Exception:
                logger.warning(
                    "Structured refinement failed – falling back to plain text",
                    exc_info=True,
                )

        raw = await self._llm.generate(
            prompt, system_prompt=REFINEMENT_SYSTEM_PROMPT
        )
        return GenerationResult(answer=raw, cited_sources=[])
