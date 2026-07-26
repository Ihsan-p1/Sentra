"""
Unit tests for ChatbotEngine's pure logic: follow-up resolution, turn recording
and claim categorisation.

ChatbotEngine.__init__ loads ML models and a database pool, so these tests build
a bare instance with object.__new__ and attach only the collaborators the methods
under test actually touch.
"""
import pytest

from chatbot.conversation_memory import ConversationMemory
from chatbot.engine import ChatbotEngine
from chatbot.language import ENGLISH, INDONESIAN, INTENT_GREETING
from config.settings import settings


@pytest.fixture
def engine():
    """A ChatbotEngine with nothing wired up except conversation memory."""
    instance = object.__new__(ChatbotEngine)
    instance.memory = ConversationMemory()
    return instance


class TestRemember:
    def test_turn_is_stored_as_user_then_assistant(self, engine):
        engine._remember("s1", "who won?", "Prabowo won.")

        history = engine.memory.get_history("s1")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "who won?"
        assert history[1]["content"] == "Prabowo won."

    def test_turns_accumulate(self, engine):
        engine._remember("s1", "q1", "a1")
        engine._remember("s1", "q2", "a2")

        assert len(engine.memory.get_history("s1")) == 4


class TestQueryRefinement:
    def test_plain_question_passes_through_unchanged(self, engine):
        query, topic = engine._handle_query_refinement("who won the 2024 election?", "s1", "default")

        assert query == "who won the 2024 election?"
        assert topic is None

    def test_strict_mode_refuses_a_two_word_query(self, engine):
        result, topic = engine._handle_query_refinement("who won", "s1", "reduce_hallucination")

        assert isinstance(result, dict)
        assert result["analysis"]["error"] == "query_too_short"
        assert result["confidence"] == 0.0
        assert topic is None

    def test_default_mode_allows_a_short_query(self, engine):
        query, _ = engine._handle_query_refinement("who won", "s1", "default")
        assert query == "who won"

    def test_explicit_topic_is_extracted_from_a_continuation(self, engine):
        query, topic = engine._handle_query_refinement(
            "tell me more about the cabinet appointments", "s1", "default"
        )

        assert topic == "the cabinet appointments"
        assert query == "the cabinet appointments"

    def test_yes_resolves_against_the_offer_in_the_last_answer(self, engine):
        # This whole branch was unreachable while memory was never written to.
        engine._remember(
            "s1",
            "who won the election?",
            "Prabowo Subianto won. Would you like to know more about the cabinet formation process?",
        )

        query, topic = engine._handle_query_refinement("yes", "s1", "default")

        assert topic == "the cabinet formation process"
        assert query == "the cabinet formation process"

    def test_continuation_without_an_offer_falls_back_to_last_real_question(self, engine):
        engine._remember("s1", "what happened to the coalition parties?", "They formed a majority bloc.")

        query, _ = engine._handle_query_refinement("tell me more", "s1", "default")

        assert query == "what happened to the coalition parties?"

    def test_follow_up_without_history_keeps_the_raw_query(self, engine):
        query, topic = engine._handle_query_refinement("yes", "fresh-session", "default")

        assert query == "yes"
        assert topic is None


class TestSmallTalkRouting:
    """
    Greetings must be answered without touching the corpus, and must never
    swallow a real question.
    """

    def test_greeting_returns_a_reply_without_sources(self, engine):
        result = engine._small_talk_response("halo", INTENT_GREETING, INDONESIAN, "default", "s1")

        assert result["intent"] == INTENT_GREETING
        assert result["sources"] == {}
        assert result["model_comparison"] == {}
        assert result["answer"].strip()

    def test_greeting_reply_matches_the_query_language(self, engine):
        indo = engine._small_talk_response("halo", INTENT_GREETING, INDONESIAN, "default", "s1")
        eng = engine._small_talk_response("hello", INTENT_GREETING, ENGLISH, "default", "s2")

        assert "Halo" in indo["answer"]
        assert "Hello" in eng["answer"]

    def test_small_talk_is_recorded_in_memory(self, engine):
        engine._small_talk_response("halo", INTENT_GREETING, INDONESIAN, "default", "s1")

        history = engine.memory.get_history("s1")
        assert [m["role"] for m in history] == ["user", "assistant"]

    def test_response_envelope_matches_a_normal_answer(self, engine):
        # The frontend reads these keys unconditionally
        result = engine._small_talk_response("hi", INTENT_GREETING, ENGLISH, "default", "s1")

        for key in ("answer", "sources", "model_comparison", "verification_summary",
                    "confidence_score", "session_id", "language"):
            assert key in result


class TestLocalisedRefusals:
    def test_short_query_refusal_follows_the_query_language(self, engine):
        indo, _ = engine._handle_query_refinement("siapa dia", "s1", "reduce_hallucination", INDONESIAN)
        eng, _ = engine._handle_query_refinement("who dat", "s2", "reduce_hallucination", ENGLISH)

        assert indo["language"] == INDONESIAN
        assert "tiga kata" in indo["answer"]
        assert "three" in eng["answer"]

    def test_claim_categories_follow_the_answer_language(self, engine):
        claims = [{"sentence": "Jokowi facilitated the transition process."}]

        assert engine._extract_claim_categories(claims, ENGLISH) == [
            "Role of President Jokowi in the process"
        ]
        assert engine._extract_claim_categories(claims, INDONESIAN) == [
            "Peran Presiden Jokowi dalam proses tersebut"
        ]

    def test_indonesian_keywords_also_match_categories(self, engine):
        claims = [{"sentence": "Keputusan kabinet diumumkan pekan lalu."}]

        assert engine._extract_claim_categories(claims, INDONESIAN) == [
            "Keputusan penunjukan kabinet"
        ]

    def test_disclaimer_language_matches_the_answer(self, engine):
        summary = {"level": "moderate"}

        indo = engine._build_final_answer("Jawaban.", summary, [], INDONESIAN)
        eng = engine._build_final_answer("Answer.", summary, [], ENGLISH)

        assert "Catatan:" in indo and "framing dan penekanan media" in indo
        assert "Note:" in eng and "media framing and emphasis" in eng

    def test_high_verification_adds_only_the_global_disclaimer(self, engine):
        answer = engine._build_final_answer("Answer.", {"level": "high"}, [], ENGLISH)

        assert "Interpretive Claims" not in answer
        assert "This analysis focuses on media framing" in answer


class TestRetrievalThresholds:
    """
    Guards on the refusal gate. The values come from
    scripts/calibrate_threshold.py; these tests stop them being nudged by feel.
    """

    # Calibration figures, measured after the corpus was re-ingested with the
    # configured embedding model
    MEASURED_OUT_OF_CORPUS_CEILING = 0.339
    MEASURED_IN_CORPUS_FLOOR = 0.329

    def test_default_threshold_rejects_everything_out_of_corpus(self):
        assert settings.RETRIEVAL_THRESHOLD > self.MEASURED_OUT_OF_CORPUS_CEILING

    def test_strict_mode_is_stricter_than_default(self):
        assert settings.RETRIEVAL_THRESHOLD_STRICT > settings.RETRIEVAL_THRESHOLD

    def test_default_threshold_is_not_needlessly_high(self):
        # Above ~0.50 the calibration set starts losing genuinely valid queries.
        assert settings.RETRIEVAL_THRESHOLD < 0.50

    def test_context_cap_is_not_tighter_than_retrieval_depth_by_much(self):
        # A cap of 3 against top_k=8 threw away the only chunk naming the 2024
        # election winner; 8 buried it. 5 is the measured sweet spot.
        assert 4 <= settings.MAX_CONTEXT_CHUNKS_PER_SOURCE <= 6

    def test_thresholds_are_valid_cosine_similarities(self):
        for value in (settings.RETRIEVAL_THRESHOLD, settings.RETRIEVAL_THRESHOLD_STRICT):
            assert 0.0 < value < 1.0


class TestClaimCategories:
    def test_keywords_map_to_a_readable_category(self, engine):
        categories = engine._extract_claim_categories(
            [{"sentence": "Jokowi facilitated the transition process."}]
        )

        assert categories == ["Role of President Jokowi in the process"]

    def test_duplicate_categories_are_collapsed(self, engine):
        categories = engine._extract_claim_categories(
            [
                {"sentence": "The coalition parties negotiated."},
                {"sentence": "Another political alliance formed."},
            ]
        )

        assert len(categories) == 1

    def test_unmatched_claims_get_a_generic_category(self, engine):
        categories = engine._extract_claim_categories([{"sentence": "Something entirely unrelated."}])

        assert categories == ["General interpretive framing statements"]

    def test_no_claims_yields_no_categories(self, engine):
        assert engine._extract_claim_categories([]) == []
