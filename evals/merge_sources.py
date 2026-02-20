"""Merge expected_citation_pattern into expected_sources as known-correct source.

The annotation script only judges what was RETRIEVED, so it may miss
sources that should be there but weren't in the top-25. This script
merges the original expected_citation_pattern (when it's specific enough,
i.e. identifies a specific article/annex) back into expected_sources.
"""

import json
import re
from pathlib import Path

DATASET_PATH = Path("evals/golden_dataset.json")

# Patterns specific enough to be ground truth (matches a specific section)
# e.g. "EUAI_Art5" or "EUAI_Art10" or "EUAI_AnnexIII" or "EUAI_Art14"
SPECIFIC_PATTERN = re.compile(
    r"^EUAI_(Art\d+|Annex[A-Z]+|Rec\d+)$"
)

# Manual overrides for questions where we know the right primary source
MANUAL_OVERRIDES: dict[str, list[str]] = {
    # Social scoring is explicitly prohibited in Article 5
    "eval_001": ["EUAI_Art5"],
    # Credit scoring → Annex III (high-risk use cases)
    "eval_011": ["EUAI_AnnexIII"],
    # Recruitment → Annex III
    "eval_020": ["EUAI_AnnexIII"],
    # Emotion recognition → Article 5 (prohibited practices)
    "eval_022": ["EUAI_Art5"],
    # Data governance → Article 10
    "eval_016": ["EUAI_Art10"],
}


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    changes = 0

    for item in dataset:
        qid = item["id"]
        old_pattern = item.get("expected_citation_pattern", "")
        sources = set(item.get("expected_sources", []))
        original = set(sources)

        # 1. Merge specific old pattern
        if old_pattern and SPECIFIC_PATTERN.match(old_pattern):
            sources.add(old_pattern)

        # 2. Apply manual overrides
        if qid in MANUAL_OVERRIDES:
            sources.update(MANUAL_OVERRIDES[qid])

        # Write back if changed
        if sources != original:
            item["expected_sources"] = sorted(sources)
            added = sources - original
            print(f"{qid}: +{added}")
            changes += 1

    if changes > 0:
        output = json.dumps(dataset, indent=2, ensure_ascii=False) + "\n"
        DATASET_PATH.write_text(output, encoding="utf-8")
        print(f"\n✅ Updated {changes} questions in {DATASET_PATH}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
