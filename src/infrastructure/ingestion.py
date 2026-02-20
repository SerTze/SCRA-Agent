"""EU AI Act ingestion pipeline."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from src.config.settings import Settings
from src.domain.exceptions import IngestionError
from src.domain.models import ChunkingStrategy, EvidenceChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_CHUNK_CHARS = 2000
_CHUNK_OVERLAP_CHARS = 200
_RECITAL_RE = re.compile(r"^\((\d+)\)\s+", re.MULTILINE)
# Legal paragraph markers – used by paragraph split mode
_PARA_MARKER_RE = re.compile(
    r"(?:^|\n)"            # start of string or newline
    r"(?:"                  # open non-capturing group
    r"\d+\.\s"             # numbered paragraph: "1. "
    r"|\([a-z]\)\s"        # lettered sub-paragraph: "(a) "
    r"|\([ivxlc]+\)\s"     # roman-numeral sub-paragraph: "(i) "
    r")",
    re.IGNORECASE,
)
_BASE_META = {
    "regulation": "EU_AI_ACT",
    "year": "2024",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_FALLBACK_EURLEX_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689"
)

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class IngestionPipeline:
    """Download, parse, and chunk the EU AI Act."""

    def __init__(
        self,
        settings: Settings,
        local_html_path: str | Path | None = None,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> None:
        self._settings = settings
        self._source_url = settings.EUR_LEX_URL
        self._local_path = Path(local_html_path) if local_html_path else None
        self._strategy = chunking_strategy or ChunkingStrategy()

    async def run(self) -> list[EvidenceChunk]:
        """Execute ingestion and return EvidenceChunks."""
        logger.info(
            "Ingestion starting (strategy=%s, mode=%s, max=%d, overlap=%d)",
            self._strategy.name,
            self._strategy.split_mode,
            self._strategy.max_chars,
            self._strategy.overlap_chars,
        )
        html = await self._download()
        chunks = self._parse_html(html)
        if not chunks:
            logger.warning("Structured parsing yielded 0 chunks – falling back")
            chunks = self._fallback_plain_text(html)

        # Alert when structured parsing degrades to fallback-quality output.
        # A healthy parse of the EU AI Act produces 100+ article chunks;
        # significantly fewer suggests the HTML structure has changed.
        article_chunks = [c for c in chunks if c.source_id.startswith("EUAI_Art")]
        if len(article_chunks) < 10:
            logger.warning(
                "INGESTION DEGRADATION: only %d article chunks produced "
                "(expected 100+). The EUR-Lex HTML structure may have changed. "
                "Review the ingestion pipeline and consider updating the parser.",
                len(article_chunks),
            )

        logger.info("Ingestion complete: %d chunks produced", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Download  (local file → CELLAR → EUR-Lex)
    # ------------------------------------------------------------------
    async def _download(self) -> str:
        """Fetch EU AI Act HTML from local file or remote sources."""
        # 1. Local file
        for path in filter(None, [self._local_path, Path("eu_ai_act.html")]):
            if path.exists():
                logger.info("Loading EU AI Act from local file: %s", path)
                return path.read_text(encoding="utf-8")

        # 2. Remote sources (settings URL first, then EUR-Lex fallback)
        download_urls = [self._source_url, _FALLBACK_EURLEX_URL]
        last_exc: Exception | None = None
        async with httpx.AsyncClient(
            timeout=90, follow_redirects=True, headers=_HTTP_HEADERS
        ) as client:
            for url in download_urls:
                try:
                    logger.info("Trying: %s", url[:80])
                    resp = await client.get(url)
                    if resp.status_code == 200 and len(resp.text) > 5000:
                        logger.info(
                            "Downloaded %d chars from %s",
                            len(resp.text),
                            url[:60],
                        )
                        return resp.text
                    logger.warning(
                        "URL %s returned status=%d, length=%d – skipping",
                        url[:60], resp.status_code, len(resp.text),
                    )
                except Exception as exc:
                    logger.warning("URL %s failed: %s", url[:60], exc)
                    last_exc = exc

        raise IngestionError(
            f"All download sources failed. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Structured HTML parsing
    # ------------------------------------------------------------------
    def _parse_html(self, html: str) -> list[EvidenceChunk]:
        """Parse HTML into articles, recitals, annexes."""
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        try:
            # Use lxml-xml for XHTML from CELLAR, fall back to lxml/html.parser
            if html.strip().startswith("<?xml") or 'xmlns=' in html[:500]:
                soup = BeautifulSoup(html, "lxml-xml")
            else:
                soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        chunks: list[EvidenceChunk] = []
        body = soup.body or soup

        # --- Articles ---
        chunks.extend(self._extract_articles(body))

        # --- Recitals ---
        chunks.extend(self._extract_recitals(body))

        # --- Annexes ---
        chunks.extend(self._extract_annexes(body))

        return chunks

    # ------------------------------------------------------------------
    # Article extraction
    # ------------------------------------------------------------------
    def _extract_articles(self, body: Tag) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        full_text = body.get_text(separator="\n")

        # Find all "Article N" headings in the full text and extract the text
        # between consecutive articles as the content for each article.
        article_starts: list[tuple[str, int]] = []
        for match in re.finditer(r"(?:^|\n)\s*(Article\s+(\d+))\b", full_text):
            art_num = match.group(2)
            article_starts.append((art_num, match.start()))

        # Deduplicate: keep only the first occurrence of each article number
        seen: set[str] = set()
        unique_starts: list[tuple[str, int]] = []
        for art_num, start in article_starts:
            if art_num not in seen:
                seen.add(art_num)
                unique_starts.append((art_num, start))

        for idx, (art_num, start) in enumerate(unique_starts):
            # End = start of next article (or +MAX_CHUNK for last)
            if idx + 1 < len(unique_starts):
                end = unique_starts[idx + 1][1]
            else:
                end = start + self._strategy.max_chars * 3

            text = full_text[start:end].strip()
            if len(text) < 30:
                continue

            art_heading = f"Article {art_num}"
            for ci, sub_chunk in enumerate(
                self._chunk_text(text, section_heading=art_heading)
            ):
                source_id = f"EUAI_Art{art_num}_Chunk{ci}"
                chunks.append(
                    EvidenceChunk(
                        content=sub_chunk,
                        source_id=source_id,
                        source_type="primary_legal",
                        metadata={
                            **_BASE_META,
                            "section_type": "article",
                            "section_number": art_num,
                            "source_url": self._source_url,
                        },
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Recital extraction
    # ------------------------------------------------------------------
    def _extract_recitals(self, body: Tag) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        full_text = body.get_text(separator="\n")

        # Scope recital extraction to the preamble (before Article 1)
        # to avoid matching parenthesized numbers inside article bodies.
        article1_pos = re.search(r"\bArticle\s+1\b", full_text)
        preamble_text = full_text[: article1_pos.start()] if article1_pos else full_text

        # Recitals are numbered (1), (2), … followed by text in the preamble
        recital_matches = list(_RECITAL_RE.finditer(preamble_text))
        for idx, match in enumerate(recital_matches):
            rec_num = match.group(1)
            start = match.start()
            end = recital_matches[idx + 1].start() if idx + 1 < len(recital_matches) else start + self._strategy.max_chars
            text = preamble_text[start:end].strip()

            if len(text) < 20:
                continue

            rec_heading = f"Recital {rec_num}"
            for ci, sub_chunk in enumerate(
                self._chunk_text(text, section_heading=rec_heading)
            ):
                source_id = f"EUAI_Rec{rec_num}_Chunk{ci}"
                chunks.append(
                    EvidenceChunk(
                        content=sub_chunk,
                        source_id=source_id,
                        source_type="primary_legal",
                        metadata={
                            **_BASE_META,
                            "section_type": "recital",
                            "section_number": rec_num,
                            "source_url": self._source_url,
                        },
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Annex extraction
    # ------------------------------------------------------------------
    def _extract_annexes(self, body: Tag) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        full_text = body.get_text(separator="\n")
        annex_starts: list[tuple[str, int]] = []
        for match in re.finditer(r"(?:^|\n)\s*(ANNEX\s+([IVXLCDM\d]+))", full_text, re.IGNORECASE):
            annex_id = match.group(2)
            annex_starts.append((annex_id, match.start()))

        seen: set[str] = set()
        unique_starts: list[tuple[str, int]] = []
        for annex_id, start in annex_starts:
            if annex_id not in seen:
                seen.add(annex_id)
                unique_starts.append((annex_id, start))

        for idx, (annex_id, start) in enumerate(unique_starts):
            if idx + 1 < len(unique_starts):
                end = unique_starts[idx + 1][1]
            else:
                end = start + self._strategy.max_chars * 3

            text = full_text[start:end].strip()
            if len(text) < 30:
                continue

            annex_heading = f"Annex {annex_id}"
            for ci, sub_chunk in enumerate(
                self._chunk_text(text, section_heading=annex_heading)
            ):
                source_id = f"EUAI_Annex{annex_id}_Chunk{ci}"
                chunks.append(
                    EvidenceChunk(
                        content=sub_chunk,
                        source_id=source_id,
                        source_type="primary_legal",
                        metadata={
                            **_BASE_META,
                            "section_type": "annex",
                            "section_number": annex_id,
                            "source_url": self._source_url,
                        },
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Fallback: plain-text page-based chunking
    # ------------------------------------------------------------------
    def _fallback_plain_text(self, html: str) -> list[EvidenceChunk]:
        """Last-resort fallback: chunk raw text by page-size blocks."""
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        text = soup.get_text(separator="\n", strip=True)
        if not text:
            logger.error("Fallback: no text extracted at all")
            return [
                EvidenceChunk(
                    content="[Ingestion fallback – no content extracted]",
                    source_id="EUAI_File_empty_Chunk0",
                    source_type="primary_legal",
                    metadata={**_BASE_META, "section_type": "fallback", "source_url": self._source_url},
                )
            ]

        chunks: list[EvidenceChunk] = []
        page_size = self._strategy.max_chars * 2
        pages = [text[i:i + page_size] for i in range(0, len(text), page_size)]
        for pi, page_text in enumerate(pages):
            for ci, sub_chunk in enumerate(
                self._chunk_text(page_text, section_heading=f"Page {pi}")
            ):
                source_id = f"EUAI_Page{pi}_Chunk{ci}"
                chunks.append(
                    EvidenceChunk(
                        content=sub_chunk,
                        source_id=source_id,
                        source_type="primary_legal",
                        metadata={
                            **_BASE_META,
                            "section_type": "fallback",
                            "source_url": self._source_url,
                        },
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Text splitting – strategy dispatcher
    # ------------------------------------------------------------------
    def _chunk_text(
        self,
        text: str,
        *,
        section_heading: str = "",
    ) -> list[str]:
        """Split text using the configured chunking strategy.

        Args:
            text: Raw text for one section (article / recital / annex).
            section_heading: e.g. "Article 5" – prepended when
                ``prepend_metadata`` is enabled.
        """
        max_chars = self._strategy.max_chars
        overlap = self._strategy.overlap_chars

        if self._strategy.split_mode == "paragraph":
            pieces = self._split_by_paragraph(text, max_chars=max_chars, overlap=overlap)
        else:
            pieces = self._split_text(text, max_chars=max_chars, overlap=overlap)

        if self._strategy.prepend_metadata and section_heading:
            pieces = [f"{section_heading}\n{p}" for p in pieces]

        return pieces

    # ------------------------------------------------------------------
    # Paragraph-level splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _split_by_paragraph(
        text: str,
        max_chars: int = _MAX_CHUNK_CHARS,
        overlap: int = _CHUNK_OVERLAP_CHARS,
    ) -> list[str]:
        """Split on legal paragraph markers: ``1.``, ``(a)``, ``(i)``, etc.

        Each numbered / lettered paragraph becomes its own chunk when it
        fits within ``max_chars``.  Adjacent small paragraphs are merged
        so chunks aren't too tiny.  If a single paragraph exceeds
        ``max_chars`` it is further split using the fixed-size splitter.
        """
        # Find split points using legal paragraph markers
        markers = list(_PARA_MARKER_RE.finditer(text))

        if not markers:
            # No legal paragraph structure – fall back to fixed splitting
            return IngestionPipeline._split_text(text, max_chars=max_chars, overlap=overlap)

        # Extract paragraphs between markers
        segments: list[str] = []
        # Text before first marker (often the article heading + intro)
        if markers[0].start() > 0:
            pre = text[: markers[0].start()].strip()
            if pre:
                segments.append(pre)

        for i, m in enumerate(markers):
            start = m.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)

        # Merge small segments so chunks aren't too tiny, split large ones
        result: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for seg in segments:
            if len(seg) > max_chars:
                # Flush what we have
                if current_parts:
                    result.append("\n".join(current_parts))
                    current_parts = []
                    current_len = 0
                # Split the oversized segment
                for piece in IngestionPipeline._split_text(seg, max_chars=max_chars, overlap=overlap):
                    result.append(piece)
                continue

            if current_len + len(seg) + 1 > max_chars:
                result.append("\n".join(current_parts))
                # Keep last segment as overlap context
                if current_parts and len(current_parts[-1]) <= overlap:
                    current_parts = [current_parts[-1]]
                    current_len = len(current_parts[0])
                else:
                    current_parts = []
                    current_len = 0

            current_parts.append(seg)
            current_len += len(seg) + 1

        if current_parts:
            result.append("\n".join(current_parts))

        return result if result else [text[:max_chars]]

    # ------------------------------------------------------------------
    # Fixed-size text splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _split_text(
        text: str,
        max_chars: int = _MAX_CHUNK_CHARS,
        overlap: int = _CHUNK_OVERLAP_CHARS,
    ) -> list[str]:
        """Split long text into <= max_chars pieces on paragraph boundaries.

        Adjacent chunks share ``overlap`` characters of trailing context
        from the previous chunk to avoid losing information at boundaries.
        """
        paragraphs = text.split("\n")
        current: list[str] = []
        current_len = 0
        result: list[str] = []

        def _flush() -> None:
            nonlocal current, current_len
            if current:
                result.append("\n".join(current))
                # Keep trailing paragraphs that fit within the overlap budget
                # so the next chunk starts with shared context.
                overlap_paras: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) + 1 > overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_len += len(p) + 1
                current = overlap_paras
                current_len = overlap_len

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If a single paragraph exceeds max_chars, split it into pieces
            if len(para) > max_chars:
                _flush()
                for i in range(0, len(para), max_chars - overlap):
                    result.append(para[i:i + max_chars])
                current = []
                current_len = 0
                continue
            if current_len + len(para) + 1 > max_chars:
                _flush()
            current.append(para)
            current_len += len(para) + 1

        if current:
            result.append("\n".join(current))
        return result if result else [text[:max_chars]]
