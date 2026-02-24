"""Groq LLM adapter – implements LLMPort via shared BaseLLMAdapter."""

from __future__ import annotations

from langchain_groq import ChatGroq

from src.config.settings import Settings
from src.domain.exceptions import AdapterError
from src.infrastructure.base_llm_adapter import BaseLLMAdapter


class GroqAdapter(BaseLLMAdapter):
    """Wraps the Groq API via langchain-groq and satisfies ``LLMPort``."""

    def __init__(self, settings: Settings) -> None:
        if not settings.GROQ_API_KEY:
            raise AdapterError(
                "GROQ_API_KEY is not configured. Set it in .env or as an environment variable."
            )
        self._settings = settings
        self._provider_name = "Groq"
        self._client = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
            timeout=30,
            max_retries=0,  # Handled by tenacity in base class
        )
        super().__init__()
