"""
Unit tests for the Model B baselines.

The A/B comparison is the academic core of this project, so the baseline has to
be deterministic: the same input must always produce the same score, otherwise
"Model A beats Model B" is not a reproducible claim.
"""
import pytest

from models.baseline.models import (
    BaselineConfidenceScorer,
    BaselineFramingAnalyzer,
    BaselineHallucinationDetector,
    get_baseline_models,
)


def chunks(n):
    return [{"text": f"chunk {i}", "similarity": 0.5} for i in range(n)]


class TestBaselineHallucinationDetector:
    @pytest.fixture
    def detector(self):
        return BaselineHallucinationDetector()

    def test_sentence_fully_present_in_source_is_supported(self, detector):
        result = detector.check_sentence(
            "Prabowo Subianto secured victory",
            ["Prabowo Subianto secured victory in the presidential election"],
        )
        assert result["is_supported"] is True
        assert result["confidence"] == pytest.approx(1.0)

    def test_sentence_absent_from_source_is_unsupported(self, detector):
        result = detector.check_sentence(
            "Indonesia announced withdrawal from ASEAN",
            ["Prabowo Subianto secured victory in the presidential election"],
        )
        assert result["is_supported"] is False

    def test_short_words_are_ignored(self, detector):
        # No word longer than 4 chars -> nothing to check, defaults to supported
        result = detector.check_sentence("a b c to of", ["anything"])
        assert result["is_supported"] is True

    def test_check_response_reports_unsupported_ratio(self, detector):
        response = (
            "Prabowo Subianto secured victory in the election. "
            "Indonesia announced immediate withdrawal from ASEAN treaties."
        )
        sources = ["Prabowo Subianto secured victory in the presidential election."]

        result = detector.check_response(response, sources)

        unsupported, total = result["unsupported_ratio"].split("/")
        assert int(total) == 2
        assert int(unsupported) == 1
        assert result["overall_score"] == pytest.approx(0.5)

    def test_markdown_headers_are_skipped(self, detector):
        result = detector.check_response("## Direct Answer", ["some source text"])
        assert result["details"] == []

    def test_response_is_deterministic(self, detector):
        response = "Prabowo Subianto secured victory in the presidential election."
        sources = ["Prabowo Subianto secured victory in the presidential election."]

        first = detector.check_response(response, sources)
        second = detector.check_response(response, sources)

        assert first == second


class TestBaselineConfidenceScorer:
    @pytest.fixture
    def scorer(self):
        return BaselineConfidenceScorer()

    def test_score_is_deterministic(self, scorer):
        retrieved = {"tempo": chunks(3), "antaranews": chunks(2)}

        scores = {scorer.score("same query", retrieved)["confidence"] for _ in range(10)}

        # Regression: the baseline used to add random noise, making the A/B
        # comparison irreproducible between runs.
        assert len(scores) == 1

    def test_more_chunks_raises_confidence(self, scorer):
        low = scorer.score("q", {"tempo": chunks(1)})["confidence"]
        high = scorer.score("q", {"tempo": chunks(6)})["confidence"]

        assert high > low

    def test_more_sources_raises_confidence(self, scorer):
        one_source = scorer.score("q", {"tempo": chunks(4)})["confidence"]
        two_sources = scorer.score("q", {"tempo": chunks(2), "antaranews": chunks(2)})["confidence"]

        assert two_sources > one_source

    def test_score_stays_within_bounds(self, scorer):
        empty = scorer.score("q", {})["confidence"]
        huge = scorer.score("q", {"tempo": chunks(500), "antaranews": chunks(500)})["confidence"]

        assert 0.2 <= empty <= 0.9
        assert 0.2 <= huge <= 0.9

    def test_empty_sources_are_not_counted(self, scorer):
        result = scorer.score("q", {"tempo": chunks(2), "antaranews": []})
        assert "1 sources" in result["reasoning"]

    def test_percent_matches_score(self, scorer):
        result = scorer.score("q", {"tempo": chunks(3)})
        assert result["confidence_percent"] == f"{int(result['confidence'] * 100)}%"


class TestBaselineFramingAnalyzer:
    def test_stopwords_and_short_words_are_excluded(self):
        analyzer = BaselineFramingAnalyzer()

        result = analyzer.analyze({"tempo": ["the cabinet cabinet cabinet and the coalition coalition"]})

        keywords = result["framing_by_media"]["tempo"]["top_keywords"]
        assert keywords[0] == "cabinet"
        assert "the" not in keywords
        assert "and" not in keywords


def test_get_baseline_models_exposes_all_three():
    models = get_baseline_models()
    assert set(models) == {"hallucination", "framing", "confidence"}
