"""OpenAI LLM adapter – implements LLMPort via shared BaseLLMAdapter."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config.settings import Settings
from src.domain.exceptions import AdapterError
from src.infrastructure.base_llm_adapter import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """Wraps the OpenAI API via langchain-openai and satisfies ``LLMPort``."""

    def __init__(self, settings: Settings) -> None:
        if not settings.OPENAI_API_KEY:
            raise AdapterError(
                "OPENAI_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )
        self._settings = settings
        self._provider_name = "OpenAI"
        self._client = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            timeout=60,
            max_retries=0,  # Handled by tenacity in base class
        )
        super().__init__()
