"""
Regression tests for defects found in the code audit.

Each class here corresponds to one bug that was live in the codebase. The bug is
described in the docstring so the test explains why it exists rather than just
asserting a value.
"""
import numpy as np
import pytest

from chatbot.language import REFUSALS, refusal
from chatbot.llm_client import LLMUnavailable
from config.settings import settings
from database.connection import database_name
from models.hallucination.detector import HARD_GATE_SIMILARITY, HallucinationDetector
from pipeline.preprocessing import ArticlePreprocessor
from rag.retrieval import MIN_CHUNK_SIMILARITY


class TestCleanTextPreservesMeaningfulSymbols:
    """
    clean_text ran a whitelist regex that omitted '%' and '&'. Six corpus
    articles contain '%', so "won 58.59% of the vote" was stored and embedded as
    "won 58.59 of the vote" - stripping the unit from exactly the figures the
    election-result articles were added to make answerable.
    """

    @pytest.fixture
    def clean(self):
        return ArticlePreprocessor().clean_text

    def test_percent_sign_survives(self, clean):
        assert "58.59%" in clean("Prabowo won 58.59% of the valid votes.")

    def test_ampersand_survives(self, clean):
        assert "&" in clean("Coverage by ANTARA & Tempo.")

    def test_slash_survives(self, clean):
        assert "2024/2025" in clean("The 2024/2025 fiscal year.")

    def test_thousands_separators_survive(self, clean):
        assert "96,214,691" in clean("He received 96,214,691 votes.")

    def test_accented_names_survive(self, clean):
        assert "José" in clean("A statement from José Ramos-Horta.")

    def test_urls_are_still_removed(self, clean):
        assert "http" not in clean("See https://example.com/article for more.")

    def test_emails_are_still_removed(self, clean):
        assert "@" not in clean("Contact editor@example.com for details.")

    def test_repeated_punctuation_is_still_collapsed(self, clean):
        assert "..." not in clean("Wait... what?")

    def test_whitespace_is_still_normalised(self, clean):
        assert clean("too    many\n\nspaces") == "too many spaces"

    def test_control_symbols_are_still_stripped(self, clean):
        assert "^" not in clean("A caret ^ and a tilde ~ here.")


class TestDatabaseNameIsDerivedFromTheUrl:
    """
    The connect log printed settings.DATABASE_NAME, a hardcoded "Sentra1". Any
    manager built with an explicit URL logged "Sentra1" while connected somewhere
    else - which is how the integration suite silently truncated the development
    database while reporting the CI one.
    """

    @pytest.mark.parametrize("url,expected", [
        ("postgresql://u:p@localhost:5432/Sentra1", "Sentra1"),
        ("postgresql://u:p@localhost:5432/sentra_ci", "sentra_ci"),
        ("postgresql://u:p@host:5432/db?sslmode=require", "db"),
        ("postgresql://u:p@host:5432/db/", "db"),
    ])
    def test_name_comes_from_the_url(self, url, expected):
        assert database_name(url) == expected

    def test_empty_url_does_not_crash(self):
        assert database_name("") == "unknown"

    def test_none_url_does_not_crash(self):
        assert database_name(None) == "unknown"

    def test_settings_no_longer_carries_a_hardcoded_name(self):
        # Guard: reintroducing it recreates the lying log line.
        assert not hasattr(settings, "DATABASE_NAME")


class TestHardGateReasonIsReported:
    """
    The hard gate set a reason string that the block below it unconditionally
    overwrote, so "Hard Gate: low semantic similarity" - the reason that actually
    decided the outcome - was never shown once.
    """

    @pytest.fixture
    def detector(self):
        """A detector whose classifier always votes 'supported'."""
        instance = HallucinationDetector()

        class AlwaysSupported:
            @staticmethod
            def predict_proba(_x):
                return np.array([[0.01, 0.99]])

        class Identity:
            @staticmethod
            def transform(x):
                return x

        instance.model = AlwaysSupported()
        instance.scaler = Identity()
        instance.is_trained = True
        return instance

    def _predict(self, detector, similarity, overlap_text="zzz"):
        # Two orthogonal-ish source vectors scaled to hit the target similarity
        sentence = np.array([1.0, 0.0])
        source = np.array([similarity, np.sqrt(max(0.0, 1 - similarity**2))])
        return detector.predict(sentence, np.array([source]), overlap_text, ["qqq"])

    def test_gate_overrides_a_confident_classifier(self, detector):
        result = self._predict(detector, similarity=0.10)

        assert result["is_supported"] is False

    def test_gate_reason_names_the_gate_and_the_floor(self, detector):
        result = self._predict(detector, similarity=0.10)

        assert "Hard gate" in result["reason"]
        assert f"{HARD_GATE_SIMILARITY:.2f}" in result["reason"]

    def test_above_the_gate_the_classifier_decides(self, detector):
        result = self._predict(detector, similarity=0.90)

        assert result["is_supported"] is True
        assert result["reason"] == ""

    def test_the_gate_constant_matches_the_documented_value(self):
        assert HARD_GATE_SIMILARITY == 0.30

    def test_the_gate_is_below_the_retrieval_threshold(self):
        # A sentence can only be checked against chunks that passed the retrieval
        # gate, so a hard gate above it would be unreachable.
        assert HARD_GATE_SIMILARITY < settings.RETRIEVAL_THRESHOLD


class TestRetrievalFloorIsNotTheRefusalGate:
    """
    A bare `if score < 0.15` sat in retrieval with the comment "Covering user
    revert". It is a per-chunk floor, not the refusal threshold, and the two were
    easy to confuse.
    """

    def test_the_floor_is_named(self):
        assert MIN_CHUNK_SIMILARITY == 0.15

    def test_the_floor_is_below_the_refusal_threshold(self):
        # If it were above, chunks could be dropped before the gate ever judged
        # them, and the gate's calibration would be measuring a different set.
        assert MIN_CHUNK_SIMILARITY < settings.RETRIEVAL_THRESHOLD


class TestLlmUnavailableCopy:
    """
    A transport failure needs its own user-facing message: it is not a refusal to
    answer, and telling the user their question was unanswerable would be wrong.
    """

    def test_both_languages_are_present(self):
        assert set(REFUSALS["llm_unavailable"]) == {"en", "id"}

    def test_it_says_the_fault_is_the_server(self):
        assert "server" in refusal("llm_unavailable", "en").lower()
        assert "server" in refusal("llm_unavailable", "id").lower()

    def test_it_states_that_sources_were_found(self):
        """
        The distinction that matters: retrieval succeeded. Wording this like the
        corpus-miss refusals would tell the user their question was unanswerable
        when in fact the answer was retrievable and only the writer was down.
        """
        assert "found relevant articles" in refusal("llm_unavailable", "en").lower()
        assert "menemukan artikel yang relevan" in refusal("llm_unavailable", "id").lower()

    def test_it_is_worded_differently_from_the_corpus_miss_refusal(self):
        for language in ("en", "id"):
            assert refusal("llm_unavailable", language) != refusal("no_sources", language)

    def test_it_tells_the_user_to_retry(self):
        assert "try again" in refusal("llm_unavailable", "en").lower()
        assert "coba lagi" in refusal("llm_unavailable", "id").lower()


class TestEngineTreatsLlmFailureAsInfrastructure:
    """
    The error text used to be returned as `raw_response`, so it was split into
    sentences, scored by the hallucination detector, given a confidence
    percentage, rendered with a full A/B panel, and written to conversation
    memory where it polluted follow-up resolution.
    """

    @pytest.fixture
    def engine(self):
        from chatbot.conversation_memory import ConversationMemory
        from chatbot.engine import ChatbotEngine

        instance = object.__new__(ChatbotEngine)
        instance.memory = ConversationMemory()
        return instance

    @pytest.fixture
    def response(self, engine):
        exc = LLMUnavailable("Connection Error: could not reach x.", "Is ollama running?")
        return engine._llm_error_response(
            exc, "en", "default", "s1", {"tempo": [{"text": "chunk", "similarity": 0.6}]}
        )

    def test_no_ab_comparison_is_fabricated(self, response):
        assert response["model_comparison"] == {}
        assert response["verification_summary"] == {}

    def test_confidence_is_not_invented(self, response):
        assert response["confidence_score"] == 0.0

    def test_the_failure_is_flagged_for_the_client(self, response):
        assert response["error"] == "llm_unavailable"

    def test_retrieved_sources_are_still_returned(self, response):
        """Retrieval succeeded; those sources are real and worth showing."""
        assert response["sources"]["tempo"]

    def test_the_underlying_error_is_surfaced(self, response):
        assert "could not reach" in response["answer"]
        assert "ollama" in response["answer"]

    def test_raw_answer_is_empty_so_nothing_downstream_verifies_it(self, response):
        assert response["raw_answer"] == ""

    def test_the_failed_turn_is_not_written_to_memory(self, engine, response):
        # A failed turn must not become the context a follow-up resolves against.
        assert engine.memory.get_history("s1") == []

    def test_the_envelope_matches_a_normal_answer(self, response):
        for key in ("answer", "sources", "model_comparison", "verification_summary",
                    "confidence_score", "session_id", "language"):
            assert key in response
