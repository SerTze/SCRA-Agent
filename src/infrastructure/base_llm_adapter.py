"""Base LLM adapter – shared logic for Groq and OpenAI providers.

Eliminates duplication between GroqAdapter and OpenAIAdapter by
extracting the common generate / generate_structured / retry logic
into a single template base class.

Includes prompt/completion audit logging and basic token-cost tracking
for operational visibility.
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.exceptions import AdapterError

logger = logging.getLogger(__name__)

# Dedicated logger for LLM audit trail (prompt/completion pairs).
# Configure separately in production (e.g. write to file or log aggregator).
_audit_logger = logging.getLogger("scra.llm_audit")


class BaseLLMAdapter:
    """Template base class for LangChain-backed LLM adapters.

    Subclasses must set ``self._client`` and ``self._provider_name`` in
    their ``__init__`` before calling ``super().__init__()``.
    """

    _client: object  # LangChain ChatModel
    _provider_name: str  # e.g. "Groq", "OpenAI"

    def __init__(self) -> None:
        self._structured_cache: dict[type, object] = {}
        self._cache_lock = asyncio.Lock()
        # Cumulative cost counters (approximate) – guarded by _usage_lock
        self._usage_lock = asyncio.Lock()
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0

    # ── Cost / token tracking ────────────────────────────────────────────

    @property
    def usage_summary(self) -> dict[str, int]:
        """Return cumulative token usage stats for monitoring."""
        return {
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    async def _track_usage(self, response: object) -> None:
        """Extract and accumulate token usage from LangChain response metadata.

        Uses an async lock to prevent lost updates under concurrent requests.
        """
        prompt_tokens = 0
        completion_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            token_usage = meta.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

        async with self._usage_lock:
            self._total_calls += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens

    def _log_interaction(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response_text: str,
        elapsed_ms: float,
        structured: bool = False,
    ) -> None:
        """Write prompt/completion pair to the audit logger."""
        _audit_logger.debug(
            "LLM call [%s] structured=%s elapsed=%.0fms | "
            "system_prompt=%.120s... | prompt=%.200s... | response=%.300s...",
            self._provider_name,
            structured,
            elapsed_ms,
            system_prompt,
            prompt,
            response_text,
        )

    # ── Public API ───────────────────────────────────────────────────────

    async def generate(self, prompt: str, *, system_prompt: str = "") -> str:
        """Send a prompt and return the text completion."""
        messages = self._build_messages(prompt, system_prompt)
        start = time.perf_counter()
        try:
            response = await self._invoke_with_retry(messages)
            elapsed_ms = (time.perf_counter() - start) * 1000
            await self._track_usage(response)
            result_text = str(response.content)
            self._log_interaction(
                prompt=prompt,
                system_prompt=system_prompt,
                response_text=result_text,
                elapsed_ms=elapsed_ms,
            )
            return result_text
        except Exception as exc:
            logger.exception("%s generation failed after retries", self._provider_name)
            raise AdapterError(
                f"{self._provider_name} generation error: {exc}"
            ) from exc

    async def generate_structured(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        schema: type[BaseModel],
    ) -> BaseModel:
        """Return a validated Pydantic object via structured output."""
        structured_llm = await self._get_structured_llm(schema)
        messages = self._build_messages(prompt, system_prompt)
        start = time.perf_counter()
        try:
            response = await self._invoke_structured_with_retry(
                structured_llm, messages
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # with include_raw=True the response is a dict:
            #   {"raw": AIMessage, "parsed": BaseModel, "parsing_error": ...}
            # Extract the raw AIMessage for token tracking and the parsed model
            # to return.  Fall back gracefully if the shape is unexpected.
            raw_msg = None
            parsed = response
            if isinstance(response, dict):
                raw_msg = response.get("raw")
                parsed = response.get("parsed", response)
                if response.get("parsing_error") is not None:
                    logger.warning(
                        "Structured output parsing error: %s",
                        response["parsing_error"],
                    )

            await self._track_usage(raw_msg if raw_msg is not None else parsed)
            self._log_interaction(
                prompt=prompt,
                system_prompt=system_prompt,
                response_text=str(parsed),
                elapsed_ms=elapsed_ms,
                structured=True,
            )
            return parsed
        except Exception as exc:
            logger.exception(
                "%s structured generation failed after retries",
                self._provider_name,
            )
            raise AdapterError(
                f"{self._provider_name} structured generation error: {exc}"
            ) from exc

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list:
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    async def _get_structured_llm(self, schema: type[BaseModel]) -> object:
        """Thread-safe lazy cache for structured-output wrappers.

        Uses ``include_raw=True`` so the raw ``AIMessage`` is available
        for token-usage tracking alongside the parsed Pydantic model.
        """
        async with self._cache_lock:
            if schema not in self._structured_cache:
                self._structured_cache[schema] = (
                    self._client.with_structured_output(schema, include_raw=True)
                )
            return self._structured_cache[schema]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: list):
        """Invoke LLM with exponential backoff retry."""
        return await self._client.ainvoke(messages)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _invoke_structured_with_retry(self, structured_llm, messages: list):
        """Invoke structured LLM with exponential backoff retry."""
        return await structured_llm.ainvoke(messages)
