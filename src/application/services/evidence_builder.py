"""Shared evidence-block builder for LLM prompts.

Centralises the logic for concatenating evidence chunks into a single
text block suitable for LLM prompts.  Used by both ``GenerationService``
and ``GradingService`` so the generator and grader always see identically
formatted evidence.
"""

from __future__ import annotations

from src.domain.models import EvidenceChunk

# Token-budget estimates (roughly 1 token ≈ 4 chars).
GENERATION_EVIDENCE_MAX_CHARS = 12_000
GRADING_EVIDENCE_MAX_CHARS = 8_000

_SEPARATOR = "\n---\n"


def build_evidence_block(
    documents: list[EvidenceChunk],
    max_chars: int = GENERATION_EVIDENCE_MAX_CHARS,
) -> str:
    """Concatenate evidence chunks, truncating at chunk boundaries.

    Instead of slicing mid-text (which could corrupt a source_id),
    adds whole chunks until the budget is exhausted.

    Parameters
    ----------
    documents : list[EvidenceChunk]
        Ordered evidence chunks to include.
    max_chars : int
        Maximum character budget for the combined block.
    """
    parts: list[str] = []
    total = 0
    for doc in documents:
        entry = f"[{doc.source_id}]:\n{doc.content}"
        added_len = len(entry) + (len(_SEPARATOR) if parts else 0)
        if total + added_len > max_chars and parts:
            parts.append("[...truncated – remaining chunks omitted]")
            break
        parts.append(entry)
        total += added_len
    return _SEPARATOR.join(parts)
