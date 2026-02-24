"""Benchmark chunking strategies and pick the best one.

Runs multiple chunking strategies against the golden retrieval eval,
ranks them, and optionally applies the winner as the new default.

Usage:
    # Run benchmark (re-ingests for each strategy)
    python -m evals.benchmark_chunking

    # Quick mode – skip strategies already benchmarked
    python -m evals.benchmark_chunking --skip-existing

    # Apply the winning strategy as the new default (updates defaults in code)
    python -m evals.benchmark_chunking --apply-winner
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config.settings import Settings
from src.domain.models import ChunkingStrategy
from src.infrastructure.chroma_adapter import ChromaAdapter
from src.infrastructure.ingestion import IngestionPipeline

from evals.eval_retrieval import RetrievalReport, run_retrieval_eval

# ---------------------------------------------------------------------------
# Strategy catalogue – ordered by expected effectiveness for legal docs
# ---------------------------------------------------------------------------

STRATEGIES: list[ChunkingStrategy] = [
    # ── Baseline: current production settings ─────────────────────────
    ChunkingStrategy(
        name="fixed_2000",
        max_chars=2000,
        overlap_chars=200,
        split_mode="fixed",
        prepend_metadata=False,
    ),
    # ── Smaller fixed chunks (better precision, more chunks) ──────────
    ChunkingStrategy(
        name="fixed_1200",
        max_chars=1200,
        overlap_chars=200,
        split_mode="fixed",
        prepend_metadata=False,
    ),
    ChunkingStrategy(
        name="fixed_800",
        max_chars=800,
        overlap_chars=150,
        split_mode="fixed",
        prepend_metadata=False,
    ),
    # ── Fixed with metadata prepend (header in every chunk) ───────────
    ChunkingStrategy(
        name="fixed_1200_meta",
        max_chars=1200,
        overlap_chars=200,
        split_mode="fixed",
        prepend_metadata=True,
    ),
    # ── Paragraph-based (respects legal structure: 1., (a), (i)) ──────
    ChunkingStrategy(
        name="para_2000",
        max_chars=2000,
        overlap_chars=200,
        split_mode="paragraph",
        prepend_metadata=False,
    ),
    ChunkingStrategy(
        name="para_1200",
        max_chars=1200,
        overlap_chars=200,
        split_mode="paragraph",
        prepend_metadata=False,
    ),
    # ── Paragraph + metadata (best theoretical for legal RAG) ─────────
    ChunkingStrategy(
        name="para_1200_meta",
        max_chars=1200,
        overlap_chars=200,
        split_mode="paragraph",
        prepend_metadata=True,
    ),
    ChunkingStrategy(
        name="para_800_meta",
        max_chars=800,
        overlap_chars=150,
        split_mode="paragraph",
        prepend_metadata=True,
    ),
]


# ---------------------------------------------------------------------------
# Scoring function – weighted combination of metrics
# ---------------------------------------------------------------------------


def compute_score(report: RetrievalReport) -> float:
    """Score a strategy.  Higher is better.

    Weights:
      - recall@5  × 4  (most important: what ends up in the LLM context)
      - recall@10 × 2  (reranker can promote these)
      - recall@25 × 1  (raw retrieval breadth)
      - hit_rate  × 1  (all questions get at least 1 match)
      - 1/rank    × 1  (lower first-match rank is better)
    """
    rank_score = (1.0 / report.mean_first_match_rank) if report.mean_first_match_rank > 0 else 0.0
    return (
        report.mean_recall_at_5 * 4.0
        + report.mean_recall_at_10 * 2.0
        + report.mean_recall_at_25 * 1.0
        + report.questions_with_match_in_top5 * 1.0
        + rank_score * 1.0
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    rank: int
    name: str
    split_mode: str
    max_chars: int
    overlap: int
    meta: bool
    chunks: int
    recall5: float
    recall10: float
    recall25: float
    hit_rate: float
    mean_rank: float
    score: float


async def run_benchmark(
    skip_existing: bool = False,
    apply_winner: bool = False,
) -> list[BenchmarkRow]:
    """Run all strategies and return ranked results."""

    settings = Settings()
    results_dir = Path("evals/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    reports: list[tuple[ChunkingStrategy, RetrievalReport, float]] = []

    print("=" * 80)
    print("CHUNKING STRATEGY BENCHMARK")
    print("=" * 80)
    print(f"Strategies to evaluate: {len(STRATEGIES)}")
    print()

    for i, strategy in enumerate(STRATEGIES, 1):
        result_path = results_dir / f"retrieval_{strategy.name}.json"

        # Skip if already computed
        if skip_existing and result_path.exists():
            print(f"[{i}/{len(STRATEGIES)}] {strategy.name}: loading cached result")
            data = json.loads(result_path.read_text(encoding="utf-8"))
            report = RetrievalReport(
                **{k: v for k, v in data.items() if k in RetrievalReport.__dataclass_fields__}
            )
            score = compute_score(report)
            reports.append((strategy, report, score))
            continue

        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(STRATEGIES)}] Strategy: {strategy.name}")
        print(
            f"  mode={strategy.split_mode}  max={strategy.max_chars}  "
            f"overlap={strategy.overlap_chars}  meta={strategy.prepend_metadata}"
        )
        print(f"{'─' * 70}")

        t0 = time.time()

        # ── Ingest ────────────────────────────────────────────────────
        collection_name = f"eval_{strategy.name}"
        eval_settings = settings.model_copy(update={"CHROMA_COLLECTION_NAME": collection_name})

        pipeline = IngestionPipeline(eval_settings, chunking_strategy=strategy)
        chunks = await pipeline.run()
        print(f"  Ingested: {len(chunks)} chunks")

        # Store in ChromaDB (clear first)
        adapter = ChromaAdapter(eval_settings)
        try:
            count = adapter._collection.count()
            if count > 0:
                adapter._client.delete_collection(collection_name)
                adapter._collection = adapter._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception:
            pass
        await adapter.add_documents(chunks)

        # ── Evaluate ──────────────────────────────────────────────────
        report = await run_retrieval_eval(
            settings=eval_settings,
            strategy=strategy,
            reingest=False,  # already ingested above
        )

        elapsed = time.time() - t0
        score = compute_score(report)

        print(
            f"  Recall@5={report.mean_recall_at_5:.0%}  "
            f"Recall@10={report.mean_recall_at_10:.0%}  "
            f"Recall@25={report.mean_recall_at_25:.0%}  "
            f"HitRate={report.questions_with_match_in_top5:.0%}  "
            f"MeanRank={report.mean_first_match_rank:.1f}"
        )
        print(f"  Score: {score:.3f}  ({elapsed:.1f}s)")

        # Save individual result
        result_path.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )

        reports.append((strategy, report, score))

    # ── Rank and display ──────────────────────────────────────────────
    reports.sort(key=lambda x: x[2], reverse=True)

    rows: list[BenchmarkRow] = []
    for rank, (strat, rep, sc) in enumerate(reports, 1):
        rows.append(
            BenchmarkRow(
                rank=rank,
                name=strat.name,
                split_mode=strat.split_mode,
                max_chars=strat.max_chars,
                overlap=strat.overlap_chars,
                meta=strat.prepend_metadata,
                chunks=rep.chunk_count,
                recall5=rep.mean_recall_at_5,
                recall10=rep.mean_recall_at_10,
                recall25=rep.mean_recall_at_25,
                hit_rate=rep.questions_with_match_in_top5,
                mean_rank=rep.mean_first_match_rank,
                score=sc,
            )
        )

    print("\n\n" + "=" * 100)
    print("BENCHMARK RESULTS – RANKED BY COMPOSITE SCORE")
    print("=" * 100)
    header = (
        f"{'#':>2}  {'Strategy':<20} {'Mode':<10} {'Max':>5} {'Ov':>4} "
        f"{'Meta':>4} {'Chunks':>6}  {'R@5':>5} {'R@10':>5} {'R@25':>5} "
        f"{'Hit%':>5} {'Rank1':>5}  {'Score':>6}"
    )
    print(header)
    print("─" * 100)
    for r in rows:
        marker = " ★" if r.rank == 1 else ""
        print(
            f"{r.rank:>2}  {r.name:<20} {r.split_mode:<10} {r.max_chars:>5} "
            f"{r.overlap:>4} {'Y' if r.meta else 'N':>4} {r.chunks:>6}  "
            f"{r.recall5:>5.0%} {r.recall10:>5.0%} {r.recall25:>5.0%} "
            f"{r.hit_rate:>5.0%} {r.mean_rank:>5.1f}  {r.score:>6.3f}{marker}"
        )
    print("=" * 100)

    winner_strat, winner_report, winner_score = reports[0]
    print(f"\n★ WINNER: {winner_strat.name} (score={winner_score:.3f})")
    print(
        f'  split_mode="{winner_strat.split_mode}", max_chars={winner_strat.max_chars}, '
        f"overlap_chars={winner_strat.overlap_chars}, prepend_metadata={winner_strat.prepend_metadata}"
    )

    # ── Save summary ──────────────────────────────────────────────────
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "winner": winner_strat.name,
        "winner_config": {
            "split_mode": winner_strat.split_mode,
            "max_chars": winner_strat.max_chars,
            "overlap_chars": winner_strat.overlap_chars,
            "prepend_metadata": winner_strat.prepend_metadata,
        },
        "winner_score": winner_score,
        "ranking": [asdict(r) for r in rows],
    }
    summary_path = results_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSummary saved to {summary_path}")

    # ── Apply winner ──────────────────────────────────────────────────
    if apply_winner:
        _apply_winner(winner_strat)

    return rows


def _apply_winner(winner: ChunkingStrategy) -> None:
    """Update the ChunkingStrategy defaults in models.py to match the winner."""
    models_path = Path("src/domain/models.py")
    content = models_path.read_text(encoding="utf-8")

    import re

    # Use a function replacement to avoid backreference interpretation issues
    def _make_replacer(value: str):
        """Return a callable that preserves the captured group and appends value."""

        def replacer(m: re.Match) -> str:
            return m.group(1) + value

        return replacer

    replacements: list[tuple[str, str]] = [
        (r'(name:\s*str\s*=\s*)"[^"]*"', f'"{winner.name}"'),
        (r"(max_chars:\s*int\s*=\s*)\d+", str(winner.max_chars)),
        (r"(overlap_chars:\s*int\s*=\s*)\d+", str(winner.overlap_chars)),
        (r'(split_mode:\s*Literal\[.*?\]\s*=\s*)"[^"]*"', f'"{winner.split_mode}"'),
        (r"(prepend_metadata:\s*bool\s*=\s*)\w+", str(winner.prepend_metadata)),
    ]

    for pattern, value in replacements:
        content, n = re.subn(pattern, _make_replacer(value), content, count=1)
        if n == 0:
            print(f"  ⚠ Could not update pattern: {pattern}")

    models_path.write_text(content, encoding="utf-8")
    print(f"\n✅ Updated ChunkingStrategy defaults in {models_path}")
    print(
        f'   name="{winner.name}", split_mode="{winner.split_mode}", '
        f"max_chars={winner.max_chars}, overlap_chars={winner.overlap_chars}, "
        f"prepend_metadata={winner.prepend_metadata}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark chunking strategies")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip strategies with existing result files",
    )
    parser.add_argument(
        "--apply-winner",
        action="store_true",
        help="Update ChunkingStrategy defaults to the winning strategy",
    )
    args = parser.parse_args()

    await run_benchmark(
        skip_existing=args.skip_existing,
        apply_winner=args.apply_winner,
    )


if __name__ == "__main__":
    asyncio.run(main())
