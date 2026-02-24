"""GradingService – grounding + compliance grading via LLM.

Uses provider-enforced structured output (``with_structured_output``) so
the LLM is *constrained* to return a valid Pydantic object.  Falls back
to the legacy ``generate()`` + JSON-parse path when the adapter does not
support ``generate_structured()``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.application.prompts import (
    COMPLIANCE_SYSTEM_PROMPT,
    COMPLIANCE_USER_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_USER_PROMPT,
)
from src.application.services.evidence_builder import (
    GRADING_EVIDENCE_MAX_CHARS,
    build_evidence_block,
)
from src.domain.exceptions import LLMResponseParsingError
from src.domain.models import ComplianceAnalysis, EvidenceChunk, GroundingResult
from src.domain.protocols import LLMPort

logger = logging.getLogger(__name__)


class GradingService:
    """Single-responsibility grading: grounding + compliance.

    Accepts an optional separate ``compliance_llm``.  When provided,
    compliance grading uses a different model from grounding grading,
    reducing the "LLM agrees with itself" bias.  Defaults to the same
    LLM for both if not specified.
    """

    def __init__(
        self,
        llm: LLMPort,
        compliance_llm: LLMPort | None = None,
    ) -> None:
        self._llm = llm
        self._compliance_llm = compliance_llm or llm
        # Check once whether the adapters support structured output
        self._supports_structured = hasattr(llm, "generate_structured")
        self._compliance_supports_structured = hasattr(self._compliance_llm, "generate_structured")

    @staticmethod
    def _build_evidence_for_grading(
        documents: list[EvidenceChunk],
        max_chars: int = GRADING_EVIDENCE_MAX_CHARS,
    ) -> str:
        """Delegate to the shared evidence builder.

        Uses the same format as GenerationService so the grader sees
        identically formatted evidence.
        """
        return build_evidence_block(documents, max_chars=max_chars)

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    async def grade_grounding(
        self,
        generation: str,
        documents: list[EvidenceChunk],
    ) -> GroundingResult:
        """Return a GroundingResult with score and reasoning."""
        if not documents or not generation:
            return GroundingResult(
                score="unknown", reasoning="No documents or generation provided."
            )

        evidence_text = self._build_evidence_for_grading(documents)
        prompt = GROUNDING_USER_PROMPT.format(evidence=evidence_text, generation=generation)

        # ── Structured output path (preferred) ────────────────────────
        if self._supports_structured:
            try:
                result: GroundingResult = await self._llm.generate_structured(
                    prompt,
                    system_prompt=GROUNDING_SYSTEM_PROMPT,
                    schema=GroundingResult,
                )
                logger.info(
                    "Grounding score (structured): %s – %s | evidence_insufficient=%s",
                    result.score,
                    result.reasoning,
                    result.evidence_insufficient,
                )
                return result
            except Exception:
                logger.warning(
                    "Structured grounding failed – falling back to legacy path",
                    exc_info=True,
                )

        # ── Legacy fallback: raw generate + JSON parse ────────────────
        try:
            raw = await self._llm.generate(
                prompt,
                system_prompt=GROUNDING_SYSTEM_PROMPT,
            )
            parsed = self._parse_json(raw)
            score = parsed.get("score", "unknown")
            reasoning = parsed.get("reasoning", "")
            if score not in ("grounded", "partial", "hallucinated"):
                return GroundingResult(score="unknown", reasoning=reasoning)
            logger.info("Grounding score: %s – %s", score, reasoning)
            return GroundingResult(score=score, reasoning=reasoning)
        except LLMResponseParsingError:
            logger.warning("Could not parse grounding response – returning 'unknown'")
            return GroundingResult(score="unknown", reasoning="Failed to parse grounding response.")
        except Exception:
            logger.exception("Grounding grading failed")
            return GroundingResult(score="unknown", reasoning="Grounding grading failed.")

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------
    async def grade_compliance(
        self,
        generation: str,
        documents: list[EvidenceChunk],
    ) -> ComplianceAnalysis:
        """Evaluate compliance and return a ComplianceAnalysis."""
        if not generation:
            return ComplianceAnalysis(
                is_compliant=False,
                risk_flags=["empty_generation"],
                reasoning=["No answer was generated."],
            )

        evidence_text = self._build_evidence_for_grading(documents)
        prompt = COMPLIANCE_USER_PROMPT.format(evidence=evidence_text, generation=generation)

        # ── Structured output path (preferred) ────────────────────────
        if self._compliance_supports_structured:
            try:
                result: ComplianceAnalysis = await self._compliance_llm.generate_structured(
                    prompt,
                    system_prompt=COMPLIANCE_SYSTEM_PROMPT,
                    schema=ComplianceAnalysis,
                )
                logger.info(
                    "Compliance (structured): compliant=%s, flags=%s",
                    result.is_compliant,
                    result.risk_flags,
                )
                return result
            except Exception:
                logger.warning(
                    "Structured compliance failed – falling back to legacy path",
                    exc_info=True,
                )

        # ── Legacy fallback: raw generate + JSON parse ────────────────
        try:
            raw = await self._compliance_llm.generate(
                prompt,
                system_prompt=COMPLIANCE_SYSTEM_PROMPT,
            )
            parsed = self._parse_json(raw)
            analysis = ComplianceAnalysis(
                is_compliant=parsed.get("is_compliant", False),
                risk_flags=parsed.get("risk_flags", []),
                reasoning=parsed.get("reasoning", []),
            )
            logger.info(
                "Compliance: compliant=%s, flags=%s", analysis.is_compliant, analysis.risk_flags
            )
            return analysis
        except LLMResponseParsingError:
            return ComplianceAnalysis(
                is_compliant=False,
                risk_flags=["parse_error"],
                reasoning=["Failed to parse compliance grading response."],
            )
        except Exception:
            logger.exception("Compliance grading failed")
            return ComplianceAnalysis(
                is_compliant=False,
                risk_flags=["grading_error"],
                reasoning=["Compliance grading encountered an error."],
            )

    # ------------------------------------------------------------------
    # JSON parsing helper (legacy fallback, fail-fast)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, fail-fast on invalid."""
        raw = raw.strip()
        # Strip markdown code fences robustly
        raw = re.sub(r"^```\w*\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseParsingError(raw, reason=str(exc)) from exc
