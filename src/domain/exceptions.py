"""Domain exceptions – no infrastructure imports."""

from __future__ import annotations


class SCRAError(Exception):
    """Base exception for all SCRA errors."""


class CitationValidationError(SCRAError):
    """Raised when citation format is invalid or citations are inconsistent."""

    def __init__(
        self,
        message: str,
        missing_inline: list[str] | None = None,
        missing_sources: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_inline = missing_inline or []
        self.missing_sources = missing_sources or []



class LatencyBudgetExceeded(SCRAError):
    """Raised when a request exceeds the 10-second latency budget."""

    def __init__(self, elapsed_seconds: float, budget_seconds: float = 10.0):
        super().__init__(
            f"Latency budget exceeded: {elapsed_seconds:.2f}s > {budget_seconds:.2f}s"
        )
        self.elapsed_seconds = elapsed_seconds
        self.budget_seconds = budget_seconds


class IngestionError(SCRAError):
    """Raised when document ingestion encounters an unrecoverable error."""


class LLMResponseParsingError(SCRAError):
    """Raised when LLM response cannot be parsed (fail-fast on invalid JSON)."""

    def __init__(self, raw_response: str, reason: str = ""):
        super().__init__(f"Failed to parse LLM response: {reason}")
        self.raw_response = raw_response
        self.reason = reason


class RetrieverError(SCRAError):
    """Raised when retrieval fails."""


class AdapterError(SCRAError):
    """Raised when an infrastructure adapter encounters an error."""
