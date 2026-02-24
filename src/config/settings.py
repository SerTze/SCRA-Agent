"""Centralised settings – loaded once, injected everywhere."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from env vars / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"  # "groq" or "openai"

    # ── Groq ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_TOKENS: int = 4096
    GROQ_GRADING_MODEL: str = ""  # If set, use a separate model for grading tasks

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: int = 4096
    # ── Cohere ────────────────────────────────────────────────────────────
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # ── ChromaDB ──────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "eu_ai_act_v2"
    EMBEDDING_MODEL: str = "cohere"  # "cohere" (API) or "default" (local ONNX MiniLM)

    # ── Tavily (web fallback) ─────────────────────────────────────────────
    TAVILY_API_KEY: str = ""

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 25
    TOP_K_FINAL: int = 5
    TOP_K_SIBLINGS: int = 2  # max sibling chunks to expand per retrieved chunk
    PRIMARY_SOURCE_BOOST: float = 1.2

    # ── Latency budget (seconds) ──────────────────────────────────────────
    LATENCY_BUDGET_SECONDS: float = 10.0

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_RPM: float = 30.0  # sustained requests per minute per client
    RATE_LIMIT_BURST: int = 10  # max burst size

    # ── Response cache ────────────────────────────────────────────────────
    CACHE_MAX_SIZE: int = 128  # max cached query results
    CACHE_TTL_SECONDS: float = 300.0  # cache entry TTL (5 minutes)
    CACHE_SIMILARITY_THRESHOLD: float = 0.90  # semantic cache cosine threshold
    # ── LangGraph ─────────────────────────────────────────────────────────
    MAX_RETRIES: int = 3

    # ── Ingestion ─────────────────────────────────────────────────────────
    EUR_LEX_URL: str = (
        "https://publications.europa.eu/resource/cellar/"
        "dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03/DOC_1"
    )

    # ── OpenTelemetry ─────────────────────────────────────────────────────
    OTEL_SERVICE_NAME: str = "scra-agent"
    OTEL_EXPORTER_ENDPOINT: str = ""

    # ── Langfuse (LLM observability) ──────────────────────────────────────
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"  # or self-hosted URL

    # ── Logging ───────────────────────────────────────────────────────────────────
    LOG_FORMAT: str = "text"  # "text" or "json"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/scra.log"  # "" to disable file logging
    LOG_FILE_MAX_BYTES: int = 10_485_760  # 10 MB per log file
    LOG_FILE_BACKUP_COUNT: int = 5  # keep 5 rotated files
    # ── Testing ───────────────────────────────────────────────────────────
    RUN_LIVE_TESTS: bool = False
