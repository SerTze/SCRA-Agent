"""Annotate golden dataset with ground-truth source patterns.

For each eval question this script:
  1. Retrieves top-25 chunks from ChromaDB
  2. Asks an LLM to judge which sources are genuinely relevant
  3. Writes an updated golden_dataset.json with:
       - "expected_sources": ["EUAI_Art5", "EUAI_Art6", "EUAI_AnnexIII"]
         (article-level patterns that SHOULD appear in retrieval results)

The annotation is at the **article / recital / annex level** (not chunk level)
because chunk numbering changes across strategies.  Each pattern is a prefix
that matches any chunk from that section (e.g. "EUAI_Art5" matches
EUAI_Art5_Chunk0, EUAI_Art5_Chunk1, …).

Usage:
    # Auto-annotate using LLM-as-judge  (requires OPENAI_API_KEY in .env)
    python -m evals.annotate_sources

    # Dry-run: just print suggestions without writing
    python -m evals.annotate_sources --dry-run

    # Use a specific model
    python -m evals.annotate_sources --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI
from src.config.settings import Settings
from src.infrastructure.chroma_adapter import ChromaAdapter

logger = logging.getLogger(__name__)

DATASET_PATH = Path("evals/golden_dataset.json")

# ── Prompt ────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a legal-information-retrieval expert.  You will be given:
  1. A user question about the EU AI Act
  2. A set of retrieved text chunks, each labelled with a source_id

Your task: identify which source_ids contain text that is **directly relevant**
to answering the question.

Rules:
- A source is relevant if its text contributes substantive information needed
  to answer the question correctly and completely.
- Include BOTH articles and their explanatory recitals if both are relevant.
- Exclude sources that are only tangentially related or merely mention a
  keyword without providing relevant legal substance.
- Return source_ids at the ARTICLE / RECITAL / ANNEX level (strip the
  _Chunk suffix).  For example, return "EUAI_Art5" not "EUAI_Art5_Chunk3".
- Deduplicate: if multiple chunks are from the same article, list it once.

Respond with ONLY a JSON array of relevant source prefixes.
Example: ["EUAI_Art5", "EUAI_Rec29", "EUAI_AnnexIII"]
If nothing is relevant, respond with: []
"""


def _strip_chunk_suffix(source_id: str) -> str:
    """EUAI_Art5_Chunk3 → EUAI_Art5"""
    return re.sub(r"_Chunk\d+$", "", source_id)


async def judge_relevance(
    client: AsyncOpenAI,
    model: str,
    question: str,
    chunks: list[dict],
) -> list[str]:
    """Ask LLM which chunk sources are relevant to the question."""
    # Build context – show source_id + truncated content
    chunk_descriptions = []
    for c in chunks:
        sid = c["source_id"]
        text = c["content"][:500]  # truncate for token budget
        chunk_descriptions.append(f"[{sid}]\n{text}\n")
    context = "\n---\n".join(chunk_descriptions)

    user_msg = f"QUESTION: {question}\n\nRETRIEVED CHUNKS:\n{context}"

    resp = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or "[]"
    # Parse – handle both {"sources": [...]} and bare [...]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            sources = parsed
        elif isinstance(parsed, dict):
            # Take the first array value
            for v in parsed.values():
                if isinstance(v, list):
                    sources = v
                    break
            else:
                sources = []
        else:
            sources = []
    except json.JSONDecodeError:
        # Try to extract array from the text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            sources = json.loads(match.group())
        else:
            logger.warning("Could not parse LLM response: %s", raw[:200])
            sources = []

    # Normalize: strip chunk suffixes, deduplicate
    normalized = sorted(set(_strip_chunk_suffix(s) for s in sources))
    return normalized


async def annotate_dataset(
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
    top_k: int = 25,
) -> None:
    """Annotate golden dataset with expected_sources via LLM-as-judge."""
    settings = Settings()
    adapter = ChromaAdapter(settings)
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total = len(dataset)

    print(f"Annotating {total} questions using {model} as judge...")
    print(
        f"ChromaDB collection: {settings.CHROMA_COLLECTION_NAME} "
        f"({adapter._collection.count()} chunks)"
    )
    print("=" * 70)

    for idx, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        topic = item.get("topic", "")

        # Skip out-of-scope questions
        if item.get("expected_citation_pattern") == "":
            item["expected_sources"] = []
            print(f"[{idx}/{total}] {qid}: OUT OF SCOPE → []")
            continue

        # Retrieve top-K chunks
        chunks = await adapter.retrieve(question, top_k=top_k)
        chunk_dicts = [{"source_id": c.source_id, "content": c.content} for c in chunks]

        # Ask LLM to judge
        relevant = await judge_relevance(client, model, question, chunk_dicts)

        # Store in dataset
        item["expected_sources"] = relevant

        # Compare with old pattern
        old_pattern = item.get("expected_citation_pattern", "?")

        print(f"[{idx}/{total}] {qid} ({topic}): old='{old_pattern}' → new={relevant}")

    print("=" * 70)

    if dry_run:
        print("DRY RUN – not writing file.  Suggested annotations above.")
        return

    # Write updated dataset
    output = json.dumps(dataset, indent=2, ensure_ascii=False) + "\n"
    DATASET_PATH.write_text(output, encoding="utf-8")
    print(f"✅ Updated {DATASET_PATH} with expected_sources")

    # Quick stats
    annotated = sum(1 for d in dataset if d.get("expected_sources"))
    empty = sum(1 for d in dataset if d.get("expected_sources") == [])
    avg_sources = sum(len(d.get("expected_sources", [])) for d in dataset) / total
    print(f"  Annotated: {annotated}, Empty/OOS: {empty}, Avg sources per Q: {avg_sources:.1f}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate golden dataset with ground-truth source patterns"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for relevance judgments (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print suggestions without writing to file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of chunks to retrieve per question (default: 25)",
    )
    args = parser.parse_args()

    await annotate_dataset(
        model=args.model,
        dry_run=args.dry_run,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    asyncio.run(main())
