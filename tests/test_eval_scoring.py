"""Unit tests for the evaluation scoring functions."""

from __future__ import annotations

import pytest
from evals.run_eval import (
    check_citation_source,
    check_citation_validity,
    score_relevance,
)


class TestScoreRelevance:
    def test_all_keywords_found(self):
        answer = "Social scoring is prohibited under Article 5."
        assert score_relevance(answer, ["prohibited", "Article 5"]) == 1.0

    def test_partial_keywords(self):
        answer = "Social scoring is not allowed."
        assert score_relevance(answer, ["prohibited", "Article 5", "scoring"]) == pytest.approx(
            1 / 3
        )

    def test_no_keywords(self):
        answer = "The weather is nice."
        assert score_relevance(answer, ["prohibited", "Article 5"]) == 0.0

    def test_empty_expected(self):
        assert score_relevance("any answer", []) == 1.0

    def test_case_insensitive(self):
        assert score_relevance("PROHIBITED practices", ["prohibited"]) == 1.0


class TestCitationValidity:
    def test_valid_citations(self):
        text = "See [EUAI_Art5_Chunk0] and [EUAI_Rec23_Chunk1]."
        assert check_citation_validity(text) is True

    def test_invalid_citation(self):
        text = "See [some random ref]."
        assert check_citation_validity(text) is False

    def test_no_citations_fails(self):
        assert check_citation_validity("No citations.") is False

    def test_web_citation_valid(self):
        text = "See [WEB_example.com_aabbccddee]."
        assert check_citation_validity(text) is True

    def test_mixed_valid_invalid_fails(self):
        text = "See [EUAI_Art5_Chunk0] and [BAD_source_ref]."
        assert check_citation_validity(text) is False


class TestCitationSource:
    def test_matches_pattern(self):
        assert check_citation_source(["EUAI_Art5_Chunk0"], "EUAI_Art5") is True

    def test_no_match(self):
        assert check_citation_source(["EUAI_Rec10_Chunk0"], "EUAI_Art5") is False

    def test_empty_sources(self):
        assert check_citation_source([], "EUAI_Art") is False

    def test_partial_match(self):
        assert check_citation_source(["EUAI_Art12_Chunk3"], "EUAI_Art") is True
