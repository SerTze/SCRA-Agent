"""Retrieval-quality evaluation – measures chunk recall across strategies.

Compares chunking/ingestion strategies by measuring how well retrieved
chunks cover the expected sources for each golden-dataset question.
No LLM calls needed – fast and cheap to run.

Usage:
    # Evaluate current (default) strategy against live ChromaDB
    python -m evals.eval_retrieval

    # Re-ingest with a named strategy, then evaluate
    python -m evals.eval_retrieval --strategy paragraph_1000 \\
        --split-mode paragraph --max-chars 1000 --overlap 200

    # Custom output
    python -m evals.eval_retrieval --output evals/results/retrieval_paragraph_1000.json

    # Compare two retrieval runs
    python -m evals.eval_retrieval --compare \\
        evals/results/retrieval_default.json \\
        evals/results/retrieval_paragraph_1000.json

Metrics:
    - recall@5:        fraction of expected source patterns found in top-5
    - recall@10:       same at top-10
    - recall@25:       same at top-25 (raw retrieval, no rerank)
    - mean_rank:       average rank of the first matching chunk
    - coverage:        how many distinct article chunks appear across all queries
    - chunk_count:     total chunks produced by this strategy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config.settings import Settings
from src.domain.models import ChunkingStrategy
from src.infrastructure.chroma_adapter import ChromaAdapter
from src.infrastructure.ingestion import IngestionPipeline

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def _strip_chunk_suffix(source_id: str) -> str:
    """EUAI_Art5_Chunk3 → EUAI_Art5"""
    import re
    return re.sub(r"_Chunk\d+$", "", source_id)


@dataclass
class QuestionRetrievalResult:
    """Retrieval metrics for a single question."""

    eval_id: str
    question: str
    topic: str
    expected_sources: list[str] = field(default_factory=list)
    # How many of the expected sources were found in top-K
    sources_found_in_top5: int = 0
    sources_found_in_top10: int = 0
    sources_found_in_top25: int = 0
    total_expected: int = 0
    total_retrieved: int = 0
    # Rank of the first matching chunk (1-indexed, 0 = not found)
    first_match_rank: int = 0
    # The actual source_ids retrieved (article-level, deduped)
    top5_sources: list[str] = field(default_factory=list)
    # Source-level recall: fraction of expected_sources found
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_25: float = 0.0
    # Which expected sources were found / missing
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass
class RetrievalReport:
    """Aggregate retrieval evaluation report."""

    timestamp: str = ""
    strategy_name: str = ""
    split_mode: str = ""
    max_chars: int = 0
    overlap_chars: int = 0
    prepend_metadata: bool = False
    chunk_count: int = 0
    total_questions: int = 0
    # Aggregate metrics
    mean_recall_at_5: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_recall_at_25: float = 0.0
    mean_first_match_rank: float = 0.0
    questions_with_match_in_top5: float = 0.0
    distinct_sources_retrieved: int = 0
    # Per question
    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

async def run_retrieval_eval(
    settings: Settings,
    strategy: ChunkingStrategy,
    dataset_path: str = "evals/golden_dataset.json",
    reingest: bool = False,
) -> RetrievalReport:
    """Run retrieval evaluation.

    Args:
        settings: App settings (ChromaDB path, API keys, etc.).
        strategy: The chunking strategy to evaluate.
        dataset_path: Path to golden dataset JSON.
        reingest: If True, re-ingest documents with the given strategy
            before evaluating.  If False, evaluate against existing
            ChromaDB contents.
    """
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

    report = RetrievalReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        strategy_name=strategy.name,
        split_mode=strategy.split_mode,
        max_chars=strategy.max_chars,
        overlap_chars=strategy.overlap_chars,
        prepend_metadata=strategy.prepend_metadata,
        total_questions=len(dataset),
    )

    # ── Re-ingest if requested ────────────────────────────────────────
    if reingest:
        print(f"Re-ingesting with strategy: {strategy.name} "
              f"(mode={strategy.split_mode}, max={strategy.max_chars}, "
              f"overlap={strategy.overlap_chars}, prepend={strategy.prepend_metadata})")

        # Use a separate collection for each strategy to avoid conflicts
        eval_settings = settings.model_copy(
            update={"CHROMA_COLLECTION_NAME": f"eval_{strategy.name}"}
        )
        pipeline = IngestionPipeline(
            eval_settings,
            chunking_strategy=strategy,
        )
        chunks = await pipeline.run()
        report.chunk_count = len(chunks)
        print(f"Produced {len(chunks)} chunks")

        # Store in ChromaDB
        adapter = ChromaAdapter(eval_settings)
        # Clear existing data in this collection
        try:
            existing = await adapter._collection.count()  # type: ignore[attr-defined]
            if existing > 0:
                adapter._client.delete_collection(eval_settings.CHROMA_COLLECTION_NAME)
                adapter._collection = adapter._client.get_or_create_collection(
                    name=eval_settings.CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception:
            pass
        added = await adapter.add_documents(chunks)
        print(f"Indexed {added} chunks in collection '{eval_settings.CHROMA_COLLECTION_NAME}'")
    else:
        eval_settings = settings
        adapter = ChromaAdapter(eval_settings)
        report.chunk_count = adapter._collection.count()
        print(f"Evaluating existing collection: {eval_settings.CHROMA_COLLECTION_NAME} "
              f"({report.chunk_count} chunks)")

    # ── Run queries ───────────────────────────────────────────────────
    results: list[QuestionRetrievalResult] = []
    all_sources: set[str] = set()

    for idx, item in enumerate(dataset, 1):
        expected_sources: list[str] = item.get("expected_sources", [])
        # Fallback: if no expected_sources, try old expected_citation_pattern
        if not expected_sources:
            old_pattern = item.get("expected_citation_pattern", "")
            if old_pattern:
                expected_sources = [old_pattern]

        qr = QuestionRetrievalResult(
            eval_id=item["id"],
            question=item["question"],
            topic=item.get("topic", ""),
            expected_sources=expected_sources,
            total_expected=len(expected_sources),
        )

        if not expected_sources:
            # Out-of-scope questions (e.g., GDPR) – skip retrieval scoring
            results.append(qr)
            continue

        # Retrieve top-25
        chunks = await adapter.retrieve(item["question"], top_k=25)
        qr.total_retrieved = len(chunks)

        # Build sets of article-level source_ids at each cutoff
        sources_at_5: set[str] = set()
        sources_at_10: set[str] = set()
        sources_at_25: set[str] = set()

        for rank, chunk in enumerate(chunks, 1):
            article_id = _strip_chunk_suffix(chunk.source_id)
            if rank <= 5:
                sources_at_5.add(article_id)
            if rank <= 10:
                sources_at_10.add(article_id)
            sources_at_25.add(article_id)
            all_sources.add(chunk.source_id)

            # Track first rank of any expected source
            if qr.first_match_rank == 0:
                if any(exp in chunk.source_id for exp in expected_sources):
                    qr.first_match_rank = rank

        # Count how many expected sources appear at each cutoff
        found_5 = [s for s in expected_sources if s in sources_at_5]
        found_10 = [s for s in expected_sources if s in sources_at_10]
        found_25 = [s for s in expected_sources if s in sources_at_25]

        qr.sources_found_in_top5 = len(found_5)
        qr.sources_found_in_top10 = len(found_10)
        qr.sources_found_in_top25 = len(found_25)
        qr.found = sorted(set(found_25))
        qr.missing = sorted(set(expected_sources) - set(found_25))

        n_exp = len(expected_sources)
        qr.recall_at_5 = len(found_5) / n_exp
        qr.recall_at_10 = len(found_10) / n_exp
        qr.recall_at_25 = len(found_25) / n_exp

        qr.top5_sources = sorted(sources_at_5)

        r5 = f"{qr.recall_at_5:.0%}"
        r25 = f"{qr.recall_at_25:.0%}"
        n_miss = len(qr.missing)
        print(
            f"[{idx}/{len(dataset)}] {item['id']}: "
            f"recall@5={r5} recall@25={r25} "
            f"({qr.sources_found_in_top25}/{n_exp} sources, "
            f"{n_miss} missing) rank1={qr.first_match_rank}"
        )
        if qr.missing:
            print(f"           missing: {qr.missing}")
        results.append(qr)

    # ── Aggregate ─────────────────────────────────────────────────────
    scored = [r for r in results if r.expected_sources]
    n = len(scored) or 1

    report.mean_recall_at_5 = sum(r.recall_at_5 for r in scored) / n
    report.mean_recall_at_10 = sum(r.recall_at_10 for r in scored) / n
    report.mean_recall_at_25 = sum(r.recall_at_25 for r in scored) / n

    ranked = [r.first_match_rank for r in scored if r.first_match_rank > 0]
    report.mean_first_match_rank = sum(ranked) / len(ranked) if ranked else 0.0
    report.questions_with_match_in_top5 = sum(
        1 for r in scored if r.sources_found_in_top5 > 0
    ) / n
    report.distinct_sources_retrieved = len(all_sources)
    report.results = [asdict(r) for r in results]

    return report


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_retrieval(baseline_path: str, new_path: str) -> bool:
    """Compare two retrieval runs.  Returns True if no regressions."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    new = json.loads(Path(new_path).read_text(encoding="utf-8"))

    metrics = [
        ("Recall@5", "mean_recall_at_5", True),
        ("Recall@10", "mean_recall_at_10", True),
        ("Recall@25", "mean_recall_at_25", True),
        ("Hit Rate (top-5)", "questions_with_match_in_top5", True),
        ("Mean First Match Rank", "mean_first_match_rank", False),
        ("Chunk Count", "chunk_count", None),  # informational only
        ("Distinct Sources", "distinct_sources_retrieved", True),
    ]

    print("=" * 72)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 72)
    print(f"  Baseline: {baseline.get('strategy_name', '?')} "
          f"(mode={baseline.get('split_mode')}, max={baseline.get('max_chars')}, "
          f"overlap={baseline.get('overlap_chars')})")
    print(f"  New:      {new.get('strategy_name', '?')} "
          f"(mode={new.get('split_mode')}, max={new.get('max_chars')}, "
          f"overlap={new.get('overlap_chars')})")
    print("-" * 72)
    print(f"  {'Metric':<25} {'Baseline':>10} {'New':>10} {'Delta':>15}")
    print("-" * 72)

    regressions = []

    for label, key, higher_is_better in metrics:
        old_val = baseline.get(key, 0)
        new_val = new.get(key, 0)

        if isinstance(old_val, float) and old_val <= 1.0:
            old_str = f"{old_val:.1%}"
            new_str = f"{new_val:.1%}"
            d = new_val - old_val
            sign = "+" if d >= 0 else ""
            arrow = ""
            if higher_is_better is not None and abs(d) > 0.01:
                if higher_is_better:
                    arrow = " ✅" if d > 0 else " ❌"
                else:
                    arrow = " ✅" if d < 0 else " ❌"
                if (higher_is_better and d < -0.05) or (not higher_is_better and d > 0.5):
                    regressions.append(f"{label}: {old_str} → {new_str}")
            delta_str = f"{sign}{d:.1%}{arrow}"
        else:
            old_str = str(old_val)
            new_str = str(new_val)
            d = new_val - old_val
            sign = "+" if d >= 0 else ""
            delta_str = f"{sign}{d}"

        print(f"  {label:<25} {old_str:>10} {new_str:>10} {delta_str:>15}")

    print("=" * 72)

    # Per-question changes
    baseline_by_id = {r["eval_id"]: r for r in baseline.get("results", [])}
    new_by_id = {r["eval_id"]: r for r in new.get("results", [])}

    improvements = []
    degradations = []
    for eval_id, old_r in baseline_by_id.items():
        new_r = new_by_id.get(eval_id)
        if not new_r or not old_r.get("expected_sources"):
            continue
        old_recall = old_r.get("recall_at_5", 0)
        new_recall = new_r.get("recall_at_5", 0)
        if new_recall > old_recall:
            improvements.append(
                f"  {eval_id}: recall@5 {old_recall:.0%} → {new_recall:.0%}"
            )
        elif new_recall < old_recall:
            degradations.append(
                f"  {eval_id}: recall@5 {old_recall:.0%} → {new_recall:.0%}"
            )

    if improvements:
        print(f"\n✅ {len(improvements)} question(s) gained top-5 hits:")
        for line in improvements:
            print(line)
    if degradations:
        print(f"\n❌ {len(degradations)} question(s) lost top-5 hits:")
        for line in degradations:
            print(line)

    if regressions:
        print(f"\n⚠ {len(regressions)} REGRESSION(S):")
        for r in regressions:
            print(f"  - {r}")
        return False
    else:
        print("\n✅ No regressions detected.")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PRESET_STRATEGIES: dict[str, ChunkingStrategy] = {
    "default": ChunkingStrategy(
        name="default",
        max_chars=2000,
        overlap_chars=200,
        split_mode="fixed",
    ),
    "small_fixed": ChunkingStrategy(
        name="small_fixed",
        max_chars=1000,
        overlap_chars=200,
        split_mode="fixed",
    ),
    "paragraph_2000": ChunkingStrategy(
        name="paragraph_2000",
        max_chars=2000,
        overlap_chars=200,
        split_mode="paragraph",
    ),
    "paragraph_1000": ChunkingStrategy(
        name="paragraph_1000",
        max_chars=1000,
        overlap_chars=200,
        split_mode="paragraph",
    ),
    "paragraph_800_meta": ChunkingStrategy(
        name="paragraph_800_meta",
        max_chars=800,
        overlap_chars=150,
        split_mode="paragraph",
        prepend_metadata=True,
    ),
}


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality across chunking strategies"
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy name (preset or custom). Presets: "
        + ", ".join(PRESET_STRATEGIES.keys()),
    )
    parser.add_argument(
        "--split-mode",
        choices=["fixed", "paragraph"],
        default=None,
        help="Override split mode",
    )
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--prepend-metadata", action="store_true", default=False)
    parser.add_argument(
        "--no-reingest",
        action="store_true",
        help="Skip re-ingestion, evaluate existing ChromaDB",
    )
    parser.add_argument("--dataset", default="evals/golden_dataset.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "NEW"),
        help="Compare two retrieval result files",
    )
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        ok = compare_retrieval(args.compare[0], args.compare[1])
        sys.exit(0 if ok else 1)

    # Build strategy
    if args.strategy and args.strategy in PRESET_STRATEGIES:
        strategy = PRESET_STRATEGIES[args.strategy]
    elif args.strategy:
        strategy = ChunkingStrategy(
            name=args.strategy,
            max_chars=args.max_chars or 2000,
            overlap_chars=args.overlap or 200,
            split_mode=args.split_mode or "fixed",
            prepend_metadata=args.prepend_metadata,
        )
    else:
        strategy = PRESET_STRATEGIES["default"]

    # Apply CLI overrides to preset
    if args.max_chars is not None:
        strategy = strategy.model_copy(update={"max_chars": args.max_chars})
    if args.overlap is not None:
        strategy = strategy.model_copy(update={"overlap_chars": args.overlap})
    if args.split_mode is not None:
        strategy = strategy.model_copy(update={"split_mode": args.split_mode})
    if args.prepend_metadata:
        strategy = strategy.model_copy(update={"prepend_metadata": True})

    settings = Settings()
    reingest = not args.no_reingest and args.strategy is not None

    report = await run_retrieval_eval(
        settings=settings,
        strategy=strategy,
        dataset_path=args.dataset,
        reingest=reingest,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Strategy:        {report.strategy_name}")
    print(f"  Split mode:      {report.split_mode}")
    print(f"  Max chars:       {report.max_chars}")
    print(f"  Overlap:         {report.overlap_chars}")
    print(f"  Prepend meta:    {report.prepend_metadata}")
    print(f"  Chunks:          {report.chunk_count}")
    print(f"  Questions:       {report.total_questions}")
    print("-" * 60)
    print(f"  Recall@5:        {report.mean_recall_at_5:.0%}")
    print(f"  Recall@10:       {report.mean_recall_at_10:.0%}")
    print(f"  Recall@25:       {report.mean_recall_at_25:.0%}")
    print(f"  Hit Rate (top5): {report.questions_with_match_in_top5:.0%}")
    print(f"  Mean 1st Rank:   {report.mean_first_match_rank:.1f}")
    print(f"  Distinct Sources:{report.distinct_sources_retrieved}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
