"""ChromaDB adapter – implements RetrieverPort and VectorStorePort."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import Settings
from src.domain.exceptions import AdapterError, RetrieverError
from src.domain.models import EvidenceChunk

logger = logging.getLogger(__name__)

# Regex to parse section base + chunk number from a source_id.
# E.g.  "EUAI_Art5_Chunk2"  →  base="EUAI_Art5", chunk_num=2
_SOURCE_ID_CHUNK_RE = re.compile(r"^(.+)_Chunk(\d+)$")


def _build_embedding_function(settings: Settings):
    """Return the embedding function configured by EMBEDDING_MODEL.

    ``"cohere"``  → Cohere embed-english-v3.0 (1024-dim, high quality)
    ``"default"`` → ChromaDB's built-in ONNX MiniLM-L6-V2 (384-dim)
    """
    model = settings.EMBEDDING_MODEL.lower()
    if model == "cohere" and settings.COHERE_API_KEY:
        from chromadb.utils.embedding_functions import CohereEmbeddingFunction

        logger.info("Using Cohere embed-english-v3.0 embeddings")
        return CohereEmbeddingFunction(
            api_key=settings.COHERE_API_KEY,
            model_name="embed-english-v3.0",
        )
    # Fallback to default ONNX MiniLM
    logger.info("Using default ONNX MiniLM-L6-V2 embeddings")
    return None  # ChromaDB uses its built-in default


class ChromaAdapter:
    """Wraps ChromaDB for vector storage and retrieval.

    Satisfies both ``RetrieverPort`` and ``VectorStorePort``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        ef = _build_embedding_function(settings)
        create_kwargs: dict = {
            "name": settings.CHROMA_COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        }
        if ef is not None:
            create_kwargs["embedding_function"] = ef
        self._collection = self._client.get_or_create_collection(**create_kwargs)

    # ── RetrieverPort ─────────────────────────────────────────────────────
    async def retrieve(self, query: str, top_k: int = 25) -> list[EvidenceChunk]:
        """Query ChromaDB and return EvidenceChunk list."""
        try:
            doc_count = await asyncio.to_thread(self._collection.count)
            if doc_count == 0:
                logger.warning("ChromaDB collection is empty – returning no results")
                return []

            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=min(top_k, doc_count),
                include=["documents", "metadatas", "distances"],
            )
            chunks: list[EvidenceChunk] = []
            if not results["documents"] or not results["documents"][0]:
                return chunks

            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [1.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances, strict=True):
                meta = meta or {}
                source_id = meta.get("source_id", "EUAI_Page0_Chunk0")
                source_type = meta.get("source_type", "primary_legal")
                # Filter metadata to string values only
                clean_meta = {k: str(v) for k, v in meta.items()}
                try:
                    chunk = EvidenceChunk(
                        content=doc,
                        source_id=source_id,
                        source_type=source_type,  # type: ignore[arg-type]
                        metadata=clean_meta,
                        relevance_score=max(0.0, 1.0 - dist),
                    )
                    chunks.append(chunk)
                except Exception:
                    logger.warning("Skipping chunk with invalid source_id: %s", source_id)
            return chunks
        except Exception as exc:
            logger.exception("ChromaDB retrieval failed")
            raise RetrieverError(f"ChromaDB retrieval error: {exc}") from exc

    # ── VectorStorePort ───────────────────────────────────────────────────
    _UPSERT_BATCH_SIZE = 15  # small batches to stay within Cohere trial limits
    _UPSERT_MAX_RETRIES = 5
    _UPSERT_BASE_DELAY = 6.0  # seconds between batches / retry base

    async def add_documents(self, chunks: list[EvidenceChunk]) -> int:
        """Insert evidence chunks into ChromaDB in rate-limited batches."""
        if not chunks:
            return 0
        try:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []

            for chunk in chunks:
                content_hash = hashlib.sha256(
                    f"{chunk.source_id}:{chunk.content}".encode()
                ).hexdigest()[:16]
                doc_id = f"{chunk.source_id}_{content_hash}"
                ids.append(doc_id)
                documents.append(chunk.content)
                meta = {**chunk.metadata}
                meta["source_id"] = chunk.source_id
                meta["source_type"] = chunk.source_type
                metadatas.append(meta)

            batch = self._UPSERT_BATCH_SIZE
            total = len(ids)
            for start in range(0, total, batch):
                end = min(start + batch, total)
                await self._upsert_with_retry(
                    ids[start:end],
                    documents[start:end],
                    metadatas[start:end],
                )
                logger.info(
                    "Upserted batch %d–%d of %d chunks",
                    start + 1,
                    end,
                    total,
                )
                if end < total:
                    await asyncio.sleep(self._UPSERT_BASE_DELAY)

            logger.info("Added %d chunks to ChromaDB", total)
            return total
        except Exception as exc:
            logger.exception("ChromaDB add_documents failed")
            raise AdapterError(f"ChromaDB ingestion error: {exc}") from exc

    async def _upsert_with_retry(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert a single batch with exponential backoff on rate limit."""
        for attempt in range(self._UPSERT_MAX_RETRIES):
            try:
                await asyncio.to_thread(
                    self._collection.upsert,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                return
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if is_rate_limit and attempt < self._UPSERT_MAX_RETRIES - 1:
                    delay = self._UPSERT_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Rate limited on upsert (attempt %d/%d) — waiting %.0fs",
                        attempt + 1,
                        self._UPSERT_MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def health_check(self) -> bool:
        """Return True if ChromaDB is reachable."""
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            return False

    async def collection_count(self) -> int:
        """Return current document count in the collection."""
        return await asyncio.to_thread(self._collection.count)

    # ── Sibling chunk expansion ───────────────────────────────────────────
    async def retrieve_siblings(
        self,
        source_ids: list[str],
        k: int = 2,
    ) -> list[EvidenceChunk]:
        """Fetch adjacent chunks for the given *source_ids*.

        For each ``<base>_Chunk<N>`` id, retrieves chunks N-k … N+k (excluding
        originals that are already in *source_ids*).  This provides surrounding
        context from the same article / section.
        """
        if not source_ids or k <= 0:
            return []

        # Parse each source_id into (base, chunk_num)
        originals: set[str] = set(source_ids)
        sibling_ids: set[str] = set()

        for sid in source_ids:
            m = _SOURCE_ID_CHUNK_RE.match(sid)
            if not m:
                continue
            base, num = m.group(1), int(m.group(2))
            for offset in range(-k, k + 1):
                neighbour_num = num + offset
                if neighbour_num < 0:
                    continue
                candidate = f"{base}_Chunk{neighbour_num}"
                if candidate not in originals:
                    sibling_ids.add(candidate)

        if not sibling_ids:
            return []

        # Query ChromaDB by metadata source_id via $in filter
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                where={"source_id": {"$in": list(sibling_ids)}},
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.warning("Sibling chunk retrieval failed", exc_info=True)
            return []

        chunks: list[EvidenceChunk] = []
        if not results or not results.get("documents"):
            return chunks

        docs = results["documents"]
        metas = results.get("metadatas") or [{}] * len(docs)

        for doc, meta in zip(docs, metas, strict=True):
            meta = meta or {}
            source_id = meta.get("source_id", "unknown")
            source_type = meta.get("source_type", "primary_legal")
            clean_meta = {k: str(v) for k, v in meta.items()}
            try:
                chunk = EvidenceChunk(
                    content=doc,
                    source_id=source_id,
                    source_type=source_type,  # type: ignore[arg-type]
                    metadata=clean_meta,
                    relevance_score=0.5,  # neutral – siblings are contextual
                )
                chunks.append(chunk)
            except Exception:
                logger.warning("Skipping invalid sibling chunk: %s", source_id)

        logger.info(
            "Sibling expansion: requested %d, found %d",
            len(sibling_ids),
            len(chunks),
        )
        return chunks
