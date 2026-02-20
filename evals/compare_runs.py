"""Compare two SCRA evaluation runs and produce a regression report.

Usage:
    python -m evals.compare_runs evals/results/baseline.json evals/results/new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _delta(old: float, new: float, higher_is_better: bool = True) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    arrow = ""
    if abs(d) > 0.01:
        if higher_is_better:
            arrow = " ✅" if d > 0 else " ❌"
        else:
            arrow = " ✅" if d < 0 else " ❌"
    return f"{sign}{d:.1%}{arrow}"


def compare(baseline_path: str, new_path: str) -> bool:
    """Compare two runs. Returns True if no regressions detected."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    new = json.loads(Path(new_path).read_text(encoding="utf-8"))

    metrics = [
        ("Pass Rate", "pass_rate", True),
        ("Mean Relevance", "mean_relevance", True),
        ("Grounding Rate", "grounding_rate", True),
        ("Compliance Rate", "compliance_rate", True),
        ("Citation Validity", "citation_validity_rate", True),
        ("Citation Source Match", "citation_source_rate", True),
        ("Mean Latency (ms)", "mean_latency_ms", False),
        ("Fallback Rate", "fallback_rate", False),
        ("Error Rate", "error_rate", False),
    ]

    print("=" * 72)
    print("SCRA REGRESSION COMPARISON")
    print("=" * 72)
    print(f"  Baseline: {baseline.get('timestamp', '?')}")
    print(f"  New:      {new.get('timestamp', '?')}")
    print("-" * 72)
    print(f"  {'Metric':<25} {'Baseline':>10} {'New':>10} {'Delta':>15}")
    print("-" * 72)

    regressions = []

    for label, key, higher_is_better in metrics:
        old_val = baseline.get(key, 0.0)
        new_val = new.get(key, 0.0)

        if key == "mean_latency_ms":
            old_str = f"{old_val:.0f}ms"
            new_str = f"{new_val:.0f}ms"
            d = new_val - old_val
            sign = "+" if d >= 0 else ""
            arrow = ""
            if abs(d) > 100:
                arrow = " ✅" if d < 0 else " ❌"
                if d > 0:
                    regressions.append(f"{label}: {old_val:.0f}ms → {new_val:.0f}ms")
            delta_str = f"{sign}{d:.0f}ms{arrow}"
        else:
            old_str = _pct(old_val)
            new_str = _pct(new_val)
            delta_str = _delta(old_val, new_val, higher_is_better)
            d = new_val - old_val
            # Flag regression if quality metric drops > 5%
            if higher_is_better and d < -0.05:
                regressions.append(f"{label}: {_pct(old_val)} → {_pct(new_val)}")
            elif not higher_is_better and d > 0.05:
                regressions.append(f"{label}: {_pct(old_val)} → {_pct(new_val)}")

        print(f"  {label:<25} {old_str:>10} {new_str:>10} {delta_str:>15}")

    print("=" * 72)

    # Per-question regressions
    baseline_by_id = {r["eval_id"]: r for r in baseline.get("results", [])}
    new_by_id = {r["eval_id"]: r for r in new.get("results", [])}

    question_regressions = []
    for eval_id, old_r in baseline_by_id.items():
        new_r = new_by_id.get(eval_id)
        if not new_r:
            continue
        # Was passing, now failing
        old_pass = (
            old_r.get("grounding_pass")
            and old_r.get("compliance_pass")
            and old_r.get("answer_relevance", 0) > 0.5
        )
        new_pass = (
            new_r.get("grounding_pass")
            and new_r.get("compliance_pass")
            and new_r.get("answer_relevance", 0) > 0.5
        )
        if old_pass and not new_pass:
            question_regressions.append(
                f"  {eval_id}: was PASS, now FAIL "
                f"(relevance {old_r.get('answer_relevance', 0):.0%}→{new_r.get('answer_relevance', 0):.0%})"
            )

    if question_regressions:
        print("\nPer-Question Regressions:")
        for line in question_regressions:
            print(line)

    if regressions:
        print(f"\n⚠ {len(regressions)} REGRESSION(S) DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        return False
    else:
        print("\n✅ No regressions detected.")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SCRA evaluation runs")
    parser.add_argument("baseline", help="Path to baseline results JSON")
    parser.add_argument("new", help="Path to new results JSON")
    args = parser.parse_args()

    ok = compare(args.baseline, args.new)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
