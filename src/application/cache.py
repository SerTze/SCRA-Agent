"""Query cache with exact-match LRU **and** semantic similarity fallback.

The exact-match path avoids re-running the pipeline for identical questions.
The semantic path catches paraphrases (e.g. "What does Article 5 prohibit?"
vs "Which AI practices are banned under Article 5?") using cosine similarity
on ONNX MiniLM embeddings — no API call required.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight local embedder (ONNX MiniLM bundled with chromadb)
# ---------------------------------------------------------------------------
_EMBEDDER = None


def _get_embedder():
    """Lazy-load chromadb's default ONNX embedding function."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            _EMBEDDER = ONNXMiniLM_L6_V2()
            logger.info("Semantic cache embedder loaded (ONNX MiniLM-L6-V2)")
        except Exception:
            logger.warning("Could not load ONNX embedder – semantic cache disabled")
    return _EMBEDDER


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class _CacheEntry:
    """Immutable cache entry with expiration timestamp."""

    value: dict
    expires_at: float
    question: str = ""
    embedding: list[float] = field(default_factory=list)


class QueryCache:
    """Async-safe LRU cache with TTL and semantic similarity fallback.

    **Fast path** – exact SHA-256 match (zero-cost lookup).
    **Slow path** – embed the query with ONNX MiniLM and scan cached
    embeddings for cosine similarity >= ``similarity_threshold``.

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries.
    ttl_seconds : float
        Time-to-live for each entry in seconds.
    similarity_threshold : float
        Minimum cosine similarity for a semantic hit (0.0–1.0).
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: float = 300.0,
        similarity_threshold: float = 0.90,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._sim_threshold = similarity_threshold
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(question: str) -> str:
        """Normalised cache key from question text."""
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _embed(self, text: str) -> list[float]:
        """Return embedding vector for *text*, or empty list on failure."""
        embedder = _get_embedder()
        if embedder is None:
            return []
        try:
            vecs = embedder([text])
            return list(vecs[0]) if vecs else []
        except Exception:
            logger.warning("Embedding failed for cache query", exc_info=True)
            return []

    async def get(self, question: str) -> dict | None:
        """Return cached result or None if miss/expired.

        Tries exact match first, then semantic similarity.
        """
        key = self._key(question)
        now = time.monotonic()

        async with self._lock:
            # --- Fast path: exact match ---
            entry = self._store.get(key)
            if entry is not None:
                if now > entry.expires_at:
                    del self._store[key]
                else:
                    self._store.move_to_end(key)
                    logger.debug("Cache EXACT HIT for hash=%s", key[:12])
                    return entry.value

            # --- Slow path: semantic similarity ---
            query_emb = self._embed(question)
            if not query_emb:
                return None

            best_score = 0.0
            best_entry: _CacheEntry | None = None
            expired_keys: list[str] = []

            for k, e in self._store.items():
                if now > e.expires_at:
                    expired_keys.append(k)
                    continue
                if not e.embedding:
                    continue
                sim = _cosine_similarity(query_emb, e.embedding)
                if sim > best_score:
                    best_score = sim
                    best_entry = e

            # Evict expired entries
            for k in expired_keys:
                del self._store[k]

            if best_entry and best_score >= self._sim_threshold:
                logger.info(
                    "Cache SEMANTIC HIT (sim=%.3f, orig=%s)",
                    best_score,
                    best_entry.question[:60],
                )
                return best_entry.value

        return None

    async def put(self, question: str, result: dict) -> None:
        """Store a result in the cache with its embedding."""
        key = self._key(question)
        emb = self._embed(question)
        async with self._lock:
            self._store[key] = _CacheEntry(
                value=result,
                expires_at=time.monotonic() + self._ttl,
                question=question,
                embedding=emb,
            )
            self._store.move_to_end(key)
            # Evict oldest if over capacity
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)
