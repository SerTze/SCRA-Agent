"""CitationService – validates and enforces the citation contract."""

from __future__ import annotations

import logging
import re

from src.domain.exceptions import CitationValidationError
from src.domain.models import CITATION_PATTERN, EvidenceChunk

logger = logging.getLogger(__name__)


class CitationService:
    """Single-responsibility service for citation validation.

    Enforces:
      - Every cited source_id actually exists in the document set.
      - Supports both structured (list-based) and legacy (regex) paths.
    """

    # ------------------------------------------------------------------
    # Structured path (preferred) – works with GenerationResult.cited_sources
    # ------------------------------------------------------------------
    @staticmethod
    def validate_structured(
        cited_sources: list[str],
        documents: list[EvidenceChunk],
    ) -> bool:
        """Validate a structured list of cited source IDs.

        Returns True if all cited IDs exist in the document set.
        Raises CitationValidationError for unknown citations.
        """
        if not cited_sources:
            # Nothing cited – caller decides whether this is acceptable
            return True

        available_ids = {doc.source_id for doc in documents}
        unknown = sorted(set(cited_sources) - available_ids)

        if unknown:
            msg = f"Citations reference unknown documents: {unknown}"
            logger.warning("Citation validation failed: %s", msg)
            raise CitationValidationError(
                msg,
                missing_inline=unknown,
                missing_sources=[],
            )

        logger.info(
            "Citation validation passed (%d sources)", len(set(cited_sources))
        )
        return True

    # ------------------------------------------------------------------
    # Legacy regex path – kept for backward compatibility / plain-text LLMs
    # ------------------------------------------------------------------
    @staticmethod
    def extract_inline_citations(text: str) -> list[str]:
        """Return all citation IDs found inline in the text body (before Sources block)."""
        sources_idx = text.find("Sources:")
        body = text[:sources_idx] if sources_idx != -1 else text
        return CITATION_PATTERN.findall(body)

    @staticmethod
    def extract_source_block_ids(text: str) -> list[str]:
        """Return citation IDs listed in the Sources: block."""
        sources_match = re.search(r"Sources:\s*\n((?:\s*-\s*\[.*?\]\s*\n?)+)", text)
        if not sources_match:
            return []
        block = sources_match.group(1)
        return CITATION_PATTERN.findall(block)

    @classmethod
    def validate(cls, generation: str, documents: list[EvidenceChunk]) -> bool:
        """Legacy regex-based validation (kept for backward compatibility)."""
        inline_ids = set(cls.extract_inline_citations(generation))
        source_ids = set(cls.extract_source_block_ids(generation))
        available_ids = {doc.source_id for doc in documents}

        errors: list[str] = []
        inline_not_in_sources = inline_ids - source_ids
        sources_not_used_inline = source_ids - inline_ids

        if inline_not_in_sources:
            errors.append(
                f"Inline citations not in Sources block: {sorted(inline_not_in_sources)}"
            )
        if sources_not_used_inline:
            logger.warning(
                "Sources block entries not used inline (non-fatal): %s",
                sorted(sources_not_used_inline),
            )

        cited_but_missing = (inline_ids | source_ids) - available_ids
        if cited_but_missing:
            errors.append(
                f"Citations reference unknown documents: {sorted(cited_but_missing)}"
            )

        if errors:
            msg = "; ".join(errors)
            logger.warning("Citation validation failed: %s", msg)
            raise CitationValidationError(
                msg,
                missing_inline=sorted(inline_not_in_sources),
                missing_sources=sorted(sources_not_used_inline),
            )

        logger.info("Citation validation passed (%d citations)", len(inline_ids))
        return True
