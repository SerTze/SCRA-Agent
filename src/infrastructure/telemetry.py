"""OpenTelemetry tracing, Langfuse LLM observability & structured logging."""

from __future__ import annotations

import json as json_module
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from src.config.settings import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json_module.dumps(log_entry)


def _make_formatter(log_format: str) -> logging.Formatter:
    """Create the appropriate formatter based on config."""
    if log_format == "json":
        return _JSONFormatter()
    return logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")


def configure_logging(settings: Settings) -> None:
    """Set up root logging with console + optional rotating file output.

    Log destinations:
    - **Console** (always): StreamHandler to stderr.
    - **File** (when ``LOG_FILE`` is set): RotatingFileHandler with
      configurable max size and backup count.  The log directory is
      created automatically if it doesn't exist.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if root.handlers:
        return  # already configured (e.g. test doubles)

    formatter = _make_formatter(settings.LOG_FORMAT)

    # ── Console handler (always) ─────────────────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── File handler (optional) ──────────────────────────────────────
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JSONFormatter() if settings.LOG_FORMAT == "json" else formatter)
        root.addHandler(file_handler)
        logger.info("File logging enabled → %s", log_path.resolve())


# ---------------------------------------------------------------------------
# Langfuse (LLM observability)
# ---------------------------------------------------------------------------
class LangfuseManager:
    """Thread-safe, lazy Langfuse callback handler.

    Replaces the previous module-level global singleton with an
    instance owned by the DI container.
    """

    def __init__(self) -> None:
        self._handler = None

    def get_callback(self, settings: Settings):
        """Return a Langfuse ``CallbackHandler``, or *None*.

        The handler is created on first call and reused thereafter.
        Returns ``None`` when Langfuse is disabled or keys are missing.
        """
        if self._handler is not None:
            return self._handler

        if not settings.LANGFUSE_ENABLED:
            return None

        if not settings.LANGFUSE_SECRET_KEY or not settings.LANGFUSE_PUBLIC_KEY:
            logger.warning(
                "LANGFUSE_ENABLED=true but secret/public keys are missing – "
                "Langfuse tracing disabled"
            )
            return None

        try:
            from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

            self._handler = LangfuseCallbackHandler(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("Langfuse tracing enabled → %s", settings.LANGFUSE_HOST)
            return self._handler
        except ImportError:
            logger.warning("langfuse package not installed – pip install langfuse")
            return None
        except Exception:
            logger.exception("Failed to initialise Langfuse callback handler")
            return None

    def flush(self) -> None:
        """Flush any pending Langfuse events (call on shutdown)."""
        if self._handler is not None:
            try:
                self._handler.flush()
            except Exception:
                logger.debug("Langfuse flush failed (non-fatal)", exc_info=True)


# Module-level convenience instance for backward-compat callers.
_langfuse_manager = LangfuseManager()


def get_langfuse_callback(settings: Settings):
    """Return a Langfuse ``CallbackHandler`` for LangChain/LangGraph, or *None*.

    Delegates to the module-level ``LangfuseManager`` instance.
    The callback handler can be passed to LangGraph's ``ainvoke(config=…)``
    to automatically trace every LLM call, chain, and retrieval step.
    """
    return _langfuse_manager.get_callback(settings)


def flush_langfuse() -> None:
    """Flush any pending Langfuse events (call on shutdown)."""
    _langfuse_manager.flush()


def configure_telemetry(settings: Settings) -> trace.Tracer:
    """Set up the OpenTelemetry tracer and return it."""
    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    exporter: SpanExporter
    if settings.OTEL_EXPORTER_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_ENDPOINT,
            )
            # Batch processor for network exporters (amortises overhead)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Using OTLP exporter → %s", settings.OTEL_EXPORTER_ENDPOINT)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed; "
                "falling back to console exporter"
            )
            exporter = ConsoleSpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        # SimpleSpanProcessor for console: no background thread means no
        # race with stdout closing during process shutdown / test teardown.
        exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)
    logger.info("OpenTelemetry configured for service: %s", settings.OTEL_SERVICE_NAME)
    return tracer
