"""Quick smoke test for the ingestion pipeline (no API keys needed)."""

import asyncio
from collections import Counter

from src.config.settings import Settings
from src.domain.models import SOURCE_ID_PATTERN
from src.infrastructure.ingestion import IngestionPipeline


async def main():
    settings = Settings(GROQ_API_KEY="x", COHERE_API_KEY="x", TAVILY_API_KEY="x")
    pipeline = IngestionPipeline(settings)

    print(f"Downloading from: {settings.EUR_LEX_URL}")
    chunks = await pipeline.run()

    print(f"\nTotal chunks produced: {len(chunks)}")

    # Breakdown by section type
    section_types = Counter(c.metadata.get("section_type", "unknown") for c in chunks)
    print(f"By section type:  {dict(section_types)}")

    source_types = Counter(c.source_type for c in chunks)
    print(f"By source type:   {dict(source_types)}")

    # Validate every source_id
    invalid = [c.source_id for c in chunks if not SOURCE_ID_PATTERN.match(c.source_id)]
    print(f"Invalid source_ids: {len(invalid)}")
    if invalid:
        print(f"  Examples: {invalid[:5]}")

    # Show 3 sample chunks
    print("\n--- Sample chunks ---")
    for c in chunks[:3]:
        preview = c.content[:150].replace("\n", " ")
        print(f"  [{c.source_id}] ({c.source_type} / {c.metadata.get('section_type', '?')})")
        print(f"    {preview}...")
        print()

    # Verify metadata fields
    has_url = sum(1 for c in chunks if c.metadata.get("source_url"))
    print(f"Chunks with source_url: {has_url}/{len(chunks)}")

    if chunks and not invalid:
        print("\n==> Ingestion verified OK")
    elif chunks:
        print(f"\n==> Ingestion produced chunks but {len(invalid)} have invalid IDs")
    else:
        print("\n==> FAILED: 0 chunks produced")


if __name__ == "__main__":
    asyncio.run(main())
