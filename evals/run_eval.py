"""SCRA Evaluation Harness – measures answer quality across the golden dataset.

Usage:
    # Against live server (must be running)
    python -m evals.run_eval

    # Point at a specific base URL (default: http://localhost:8000)
    python -m evals.run_eval --url http://localhost:8000

    # If upstream services rate-limit (e.g., Cohere trial keys), add pacing
    python -m evals.run_eval --delay 10

    # Label the run with a model name (for reporting/compare_runs)
    python -m evals.run_eval --model "gpt-4o-mini"

    # Save results to file for comparison
    python -m evals.run_eval --output evals/results/run_2026-02-11.json

    # Compare two runs
    python -m evals.compare_runs evals/results/baseline.json evals/results/new.json

Metrics computed per question:
    - answer_relevance:  % of expected keywords found in the answer
    - citation_valid:    all inline citations match the allowed regex
    - citation_sourced:  citations reference the expected source pattern
    - grounding_pass:    grounding_score == "grounded"
    - compliance_pass:   is_compliant == True
    - latency_ms:        end-to-end time
    - no_fallback:       answered from primary corpus (no web search)

Aggregate metrics:
    - mean_relevance, mean_latency
    - grounding_rate, compliance_rate, citation_rate
    - fallback_rate
    - pass_rate (grounded + compliant + relevant > 0.5)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from src.domain.models import CITATION_PATTERN

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Evaluation result for a single question."""

    eval_id: str
    question: str
    topic: str
    difficulty: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    grounding_score: str = "unknown"
    is_compliant: bool = False
    latency_ms: float = 0.0
    fallback_used: bool = False
    # Computed metrics
    answer_relevance: float = 0.0
    citation_valid: bool = False
    citation_sourced: bool = False
    grounding_pass: bool = False
    compliance_pass: bool = False
    no_fallback: bool = True
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    timestamp: str = ""
    model: str = ""
    total_questions: int = 0
    # Aggregate scores
    mean_relevance: float = 0.0
    mean_latency_ms: float = 0.0
    grounding_rate: float = 0.0
    compliance_rate: float = 0.0
    citation_validity_rate: float = 0.0
    citation_source_rate: float = 0.0
    fallback_rate: float = 0.0
    pass_rate: float = 0.0
    error_rate: float = 0.0
    # Per-question detail
    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_relevance(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def check_citation_validity(answer: str) -> bool:
    """All citation-like inline references match the allowed regex pattern.

    Only bracket-enclosed text containing an underscore is treated as a
    citation attempt – this avoids false failures on things like ``[1]``,
    ``[emphasis added]``, etc.
    """
    raw_brackets = re.findall(r"\[([^\]]+)\]", answer)
    if not raw_brackets:
        return False  # No citations at all is a failure
    # Filter to citation-like references (contain underscore)
    citation_attempts = [ref for ref in raw_brackets if "_" in ref]
    if not citation_attempts:
        return False  # No citation-like references found
    # Every citation attempt must match the allowed pattern
    for ref in citation_attempts:
        full = f"[{ref}]"
        if not CITATION_PATTERN.match(full):
            return False
    return True


def check_citation_source(sources: list[str], expected_pattern: str) -> bool:
    """At least one source matches the expected citation pattern."""
    if not sources or not expected_pattern:
        return False
    return any(expected_pattern in s for s in sources)


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------


async def run_single_eval(
    client: httpx.AsyncClient,
    base_url: str,
    eval_item: dict,
) -> QuestionResult:
    """Send a question to the SCRA API and score the result."""
    result = QuestionResult(
        eval_id=eval_item["id"],
        question=eval_item["question"],
        topic=eval_item.get("topic", ""),
        difficulty=eval_item.get("difficulty", ""),
    )

    try:
        start = time.perf_counter()
        # Retry on 429 / 500 (upstream rate-limits)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            resp = await client.post(
                f"{base_url}/query",
                json={"question": eval_item["question"]},
                timeout=180.0,
            )
            if resp.status_code in (429, 500) and attempt < max_attempts:
                wait = 15 * attempt  # 15s, 30s
                print(f"         ⏳ {resp.status_code} – retry {attempt}/{max_attempts} in {wait}s")
                await asyncio.sleep(wait)
                continue
            break
        elapsed = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        result.answer = data.get("answer", "")
        result.sources = data.get("sources", [])
        result.grounding_score = data.get("grounding_score", "unknown")
        result.latency_ms = data.get("latency_ms", elapsed)
        result.fallback_used = data.get("fallback_used", False)

        compliance = data.get("compliance")
        if compliance:
            result.is_compliant = compliance.get("is_compliant", False)

        # Score
        result.answer_relevance = score_relevance(
            result.answer, eval_item.get("expected_answer_contains", [])
        )
        result.citation_valid = check_citation_validity(result.answer)
        result.citation_sourced = check_citation_source(
            result.sources, eval_item.get("expected_citation_pattern", "")
        )
        result.grounding_pass = result.grounding_score == "grounded"
        result.compliance_pass = result.is_compliant
        result.no_fallback = not result.fallback_used

    except Exception as exc:
        result.error = str(exc)

    return result


async def run_evaluation(
    base_url: str = "http://localhost:8000",
    dataset_path: str = "evals/golden_dataset.json",
    model: str = "unknown",
    delay: float = 8.0,
) -> EvalReport:
    """Run the full evaluation suite and return an EvalReport.

    Args:
        delay: Seconds to wait between requests (avoids upstream rate-limits).
    """
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    report = EvalReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model=model,
        total_questions=len(dataset),
    )

    results: list[QuestionResult] = []
    async with httpx.AsyncClient() as client:
        for idx, item in enumerate(dataset, 1):
            print(f"[{idx}/{len(dataset)}] {item['id']}: {item['question'][:60]}...")
            qr = await run_single_eval(client, base_url, item)
            status = "PASS" if qr.grounding_pass and qr.compliance_pass else "FAIL"
            if qr.error:
                status = "ERROR"
            print(
                f"         → {status} | relevance={qr.answer_relevance:.0%} "
                f"grounded={qr.grounding_pass} compliant={qr.compliance_pass} "
                f"latency={qr.latency_ms:.0f}ms"
            )
            results.append(qr)
            # Rate-limit pacing (Cohere trial: 10 req/min)
            if idx < len(dataset):
                await asyncio.sleep(delay)

    # Aggregate
    n = len(results)
    errors = [r for r in results if r.error]
    valid = [r for r in results if not r.error]
    nv = len(valid) or 1

    report.mean_relevance = sum(r.answer_relevance for r in valid) / nv
    report.mean_latency_ms = sum(r.latency_ms for r in valid) / nv
    report.grounding_rate = sum(1 for r in valid if r.grounding_pass) / nv
    report.compliance_rate = sum(1 for r in valid if r.compliance_pass) / nv
    report.citation_validity_rate = sum(1 for r in valid if r.citation_valid) / nv
    report.citation_source_rate = sum(1 for r in valid if r.citation_sourced) / nv
    report.fallback_rate = sum(1 for r in valid if r.fallback_used) / nv
    report.error_rate = len(errors) / n if n else 0.0

    # Pass = grounded + compliant + relevance > 50%
    passing = sum(
        1 for r in valid if r.grounding_pass and r.compliance_pass and r.answer_relevance > 0.5
    )
    report.pass_rate = passing / nv

    report.results = [asdict(r) for r in results]

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="SCRA Evaluation Harness")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the SCRA API")
    parser.add_argument(
        "--dataset", default="evals/golden_dataset.json", help="Path to golden dataset"
    )
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--model", default="unknown", help="Model name for report labeling")
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help="Seconds between requests to respect upstream rate-limits (default: 8)",
    )
    args = parser.parse_args()

    report = await run_evaluation(
        base_url=args.url,
        dataset_path=args.dataset,
        model=args.model,
        delay=args.delay,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SCRA EVALUATION REPORT")
    print("=" * 60)
    print(f"  Timestamp:              {report.timestamp}")
    print(f"  Questions:              {report.total_questions}")
    print(f"  Pass Rate:              {report.pass_rate:.0%}")
    print(f"  Mean Relevance:         {report.mean_relevance:.0%}")
    print(f"  Grounding Rate:         {report.grounding_rate:.0%}")
    print(f"  Compliance Rate:        {report.compliance_rate:.0%}")
    print(f"  Citation Validity:      {report.citation_validity_rate:.0%}")
    print(f"  Citation Source Match:  {report.citation_source_rate:.0%}")
    print(f"  Mean Latency:           {report.mean_latency_ms:.0f}ms")
    print(f"  Fallback Rate:          {report.fallback_rate:.0%}")
    print(f"  Error Rate:             {report.error_rate:.0%}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.output}")

    # Exit with failure if pass rate < 70%
    if report.pass_rate < 0.7:
        print(f"\n⚠ BELOW THRESHOLD: pass_rate={report.pass_rate:.0%} < 70%")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
