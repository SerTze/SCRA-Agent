# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for lxml / chromadb (build-time only)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system libraries (lxml, chromadb)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libxml2 libxslt1.1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY evals/ evals/
COPY pyproject.toml .
COPY requirements.txt .

# Create non-root user
RUN useradd --create-home appuser && \
    mkdir -p /app/chroma_data && \
    chown -R appuser:appuser /app
USER appuser

# ChromaDB persistence volume
VOLUME /app/chroma_data

# Default env vars (override via .env or docker-compose)
ENV CHROMA_PERSIST_DIR=/app/chroma_data \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
