"""
Unit tests for language detection, intent classification, and localised copy.

These decide whether a message reaches the RAG pipeline at all and what language
the answer comes back in, so both the happy path and the misrouting risk are
covered: a research question must never be mistaken for small talk.
"""
import pytest

from chatbot.language import (
    CLAIM_CATEGORIES,
    DISCLAIMERS,
    ENGLISH,
    HEADINGS,
    HELP_PATTERNS,
    HELP_TOPICS,
    INDONESIAN,
    INTENT_FAREWELL,
    INTENT_GREETING,
    INTENT_HELP,
    INTENT_QUESTION,
    INTENT_THANKS,
    LANGUAGE_NAMES,
    REFUSALS,
    SMALL_TALK,
    claim_category,
    classify_intent,
    detect_help_topic,
    detect_language,
    disclaimer,
    headings,
    help_reply,
    refusal,
    small_talk_reply,
)


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text",
        [
            "siapa itu prabowo subianto",
            "bagaimana media meliput pemilu",
            "apa yang terjadi setelah pemilu 2024",
            "tolong jelaskan putusan mahkamah konstitusi",
            "kenapa gibran dipilih jadi wakil presiden",
            "berapa banyak pemilih yang hadir",
        ],
    )
    def test_indonesian_questions_are_detected(self, text):
        assert detect_language(text) == INDONESIAN

    @pytest.mark.parametrize(
        "text",
        [
            "who is prabowo subianto",
            "how did the media cover the election",
            "what happened after the 2024 election",
            "please explain the constitutional court ruling",
            "compare the coverage between outlets",
        ],
    )
    def test_english_questions_are_detected(self, text):
        assert detect_language(text) == ENGLISH

    def test_empty_input_defaults_to_english(self):
        assert detect_language("") == ENGLISH
        assert detect_language("   ") == ENGLISH

    def test_bare_keywords_default_to_english(self):
        # No function words either way; the corpus and prompts are English, so
        # English is the safer default for a keyword-style query.
        assert detect_language("prabowo cabinet") == ENGLISH

    def test_indonesian_affixes_break_a_tie(self):
        # No marker words, but "pemilihan"/"kebijakan" are unmistakably Indonesian
        assert detect_language("pemilihan kebijakan") == INDONESIAN

    def test_detection_is_case_insensitive(self):
        assert detect_language("SIAPA ITU PRABOWO") == INDONESIAN
        assert detect_language("WHO IS PRABOWO") == ENGLISH

    def test_punctuation_does_not_break_detection(self):
        assert detect_language("apa kabar, siapa presiden?!") == INDONESIAN

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("halo", INDONESIAN),
            ("hai", INDONESIAN),
            ("selamat pagi", INDONESIAN),
            ("assalamualaikum", INDONESIAN),
            ("permisi", INDONESIAN),
            ("hello", ENGLISH),
            ("hi", ENGLISH),
            ("hey", ENGLISH),
            ("good morning", ENGLISH),
        ],
    )
    def test_bare_greetings_carry_their_own_language(self, text, expected):
        # Regression: greetings contain no function words, so every one of them
        # used to fall through to the English default - "halo" included.
        assert detect_language(text) == expected

    def test_every_language_has_a_display_name(self):
        for code in (ENGLISH, INDONESIAN):
            assert LANGUAGE_NAMES[code]


class TestClassifyIntent:
    @pytest.mark.parametrize(
        "text",
        ["halo", "hai", "hi", "hello", "Hai!", "selamat pagi", "good morning",
         "assalamualaikum", "apa kabar", "how are you", "siapa kamu", "who are you"],
    )
    def test_greetings_are_detected(self, text):
        assert classify_intent(text) == INTENT_GREETING

    @pytest.mark.parametrize("text", ["terima kasih", "makasih", "thanks", "thank you"])
    def test_thanks_is_detected(self, text):
        assert classify_intent(text) == INTENT_THANKS

    @pytest.mark.parametrize("text", ["bye", "goodbye", "sampai jumpa", "dadah"])
    def test_farewell_is_detected(self, text):
        assert classify_intent(text) == INTENT_FAREWELL

    @pytest.mark.parametrize(
        "text",
        [
            "siapa itu prabowo subianto",
            "who won the 2024 indonesian election",
            "bagaimana tempo meliput pembentukan kabinet",
            "compare framing between antaranews and tempo",
            "apa putusan mahkamah konstitusi",
            "tell me more",
            "prabowo cabinet appointments",
        ],
    )
    def test_real_questions_are_not_diverted(self, text):
        # The expensive failure mode: a research question silently skipping RAG.
        assert classify_intent(text) == INTENT_QUESTION

    def test_greeting_word_inside_a_real_question_is_not_small_talk(self):
        assert classify_intent("hi, what did the constitutional court decide") == INTENT_QUESTION

    def test_thanks_inside_a_long_message_is_not_small_talk(self):
        assert (
            classify_intent("thanks, now explain how tempo framed the cabinet reshuffle")
            == INTENT_QUESTION
        )

    def test_empty_message_is_treated_as_a_question(self):
        assert classify_intent("") == INTENT_QUESTION


class TestHelpIntent:
    @pytest.mark.parametrize(
        "text,topic",
        [
            ("apa fungsi dari analysis results?", "analysis_results"),
            ("what is analysis results", "analysis_results"),
            ("apa bedanya model a dan model b", "model_ab"),
            ("explain model a vs model b", "model_ab"),
            ("bagaimana cara deteksi halusinasi bekerja", "hallucination"),
            ("explain the hallucination detection", "hallucination"),
            ("apa itu framing keywords", "framing"),
            ("what does style match mean", "framing"),
            ("apa arti confidence score", "confidence"),
            ("apa itu reduce hallucination", "modes"),
            ("apa saja fitur project ini", "overview"),
            ("what features does this app have", "overview"),
            ("apa itu sentra", "overview"),
        ],
    )
    def test_app_questions_route_to_the_right_topic(self, text, topic):
        assert detect_help_topic(text) == topic
        assert classify_intent(text) == INTENT_HELP

    @pytest.mark.parametrize(
        "text",
        [
            # The expensive failure: a research question routed to a canned help
            # reply would silently skip retrieval entirely.
            "how did the outlets frame the election result",
            "bagaimana media membingkai hasil pemilu",
            "compare framing between tempo and antaranews",
            "siapa itu prabowo subianto",
            "what did the constitutional court decide",
            "apa yang terjadi setelah pemilu 2024",
            "prabowo cabinet appointments",
            "tell me more about the coalition",
        ],
    )
    def test_research_questions_are_never_treated_as_help(self, text):
        assert detect_help_topic(text) is None
        assert classify_intent(text) == INTENT_QUESTION

    def test_framing_alone_is_not_the_framing_panel(self):
        # "framing" appears in real questions; only the UI label routes to help
        assert detect_help_topic("how does tempo use framing") is None
        assert detect_help_topic("apa itu framing keywords") == "framing"

    def test_app_mention_without_a_known_panel_gives_the_overview(self):
        assert detect_help_topic("apa yang bisa aplikasi ini lakukan") == "overview"

    def test_empty_message_has_no_help_topic(self):
        assert detect_help_topic("") is None

    @pytest.mark.parametrize("topic", list(HELP_TOPICS))
    @pytest.mark.parametrize("language", [ENGLISH, INDONESIAN])
    def test_every_topic_exists_in_both_languages(self, topic, language):
        text = help_reply(topic, language)
        assert len(text) > 200, "help text should actually explain something"

    @pytest.mark.parametrize("topic", list(HELP_TOPICS))
    def test_help_text_avoids_unexplained_jargon(self, topic):
        # These terms may appear, but only alongside a plain-language gloss.
        for language in (ENGLISH, INDONESIAN):
            text = help_reply(topic, language).lower()
            if "cosine" in text or "embedding" in text:
                pytest.fail(f"{topic}/{language} exposes raw jargon to the user")

    def test_every_help_pattern_maps_to_a_real_topic(self):
        for _phrases, topic in HELP_PATTERNS:
            assert topic in HELP_TOPICS, f"unknown topic: {topic}"

    def test_overview_lists_the_main_features(self):
        for language in (ENGLISH, INDONESIAN):
            text = help_reply("overview", language).lower()
            for expected in ("framing", "confidence", "model"):
                assert expected in text


class TestLocalisedCopy:
    @pytest.mark.parametrize("intent", [INTENT_GREETING, INTENT_THANKS, INTENT_FAREWELL])
    @pytest.mark.parametrize("language", [ENGLISH, INDONESIAN])
    def test_small_talk_exists_in_both_languages(self, intent, language):
        assert small_talk_reply(intent, language).strip()

    def test_greeting_differs_by_language(self):
        assert small_talk_reply(INTENT_GREETING, ENGLISH) != small_talk_reply(
            INTENT_GREETING, INDONESIAN
        )

    def test_greeting_states_what_the_corpus_covers(self):
        # A greeting is the one place to set expectations about scope.
        for language in (ENGLISH, INDONESIAN):
            reply = small_talk_reply(INTENT_GREETING, language)
            assert "ANTARANEWS" in reply and "TEMPO" in reply

    @pytest.mark.parametrize("kind", list(REFUSALS))
    @pytest.mark.parametrize("language", [ENGLISH, INDONESIAN])
    def test_every_refusal_is_translated(self, kind, language):
        assert refusal(kind, language).strip()

    def test_refusal_explains_what_to_do_next(self):
        # Regression: strict mode used to return the bare "Insufficient source coverage."
        message = refusal("low_similarity_strict", ENGLISH)
        assert len(message) > 60
        assert "Reduce Hallucination" in message

    @pytest.mark.parametrize("kind", list(DISCLAIMERS))
    @pytest.mark.parametrize("language", [ENGLISH, INDONESIAN])
    def test_every_disclaimer_is_translated(self, kind, language):
        assert disclaimer(kind, language).strip()

    @pytest.mark.parametrize("key", list(CLAIM_CATEGORIES))
    @pytest.mark.parametrize("language", [ENGLISH, INDONESIAN])
    def test_every_claim_category_is_translated(self, key, language):
        assert claim_category(key, language).strip()

    def test_unknown_language_falls_back_to_english(self):
        assert refusal("no_sources", "fr") == refusal("no_sources", ENGLISH)
        assert small_talk_reply(INTENT_GREETING, "fr") == small_talk_reply(INTENT_GREETING, ENGLISH)
        assert disclaimer("academic", "fr") == disclaimer("academic", ENGLISH)

    def test_translation_tables_have_no_missing_entries(self):
        for table in (SMALL_TALK, REFUSALS, DISCLAIMERS, CLAIM_CATEGORIES, HEADINGS, HELP_TOPICS):
            for key, translations in table.items():
                assert set(translations) == {ENGLISH, INDONESIAN}, f"incomplete: {key}"


class TestHeadings:
    def test_every_heading_key_is_prefixed_for_prompt_formatting(self):
        assert all(key.startswith("h_") for key in headings(ENGLISH))

    def test_heading_set_is_identical_across_languages(self):
        assert set(headings(ENGLISH)) == set(headings(INDONESIAN))

    def test_headings_actually_differ_between_languages(self):
        assert headings(ENGLISH)["h_summary"] == "Summary"
        assert headings(INDONESIAN)["h_summary"] == "Ringkasan"

    def test_no_heading_is_empty(self):
        for language in (ENGLISH, INDONESIAN):
            assert all(value.strip() for value in headings(language).values())

    def test_unknown_language_falls_back_to_english_headings(self):
        assert headings("fr") == headings(ENGLISH)

    def test_headings_cover_every_placeholder_in_the_prompts(self):
        # A missing key would raise KeyError at request time, not at import.
        import re as _re

        from chatbot.prompts import FRAMING_ANALYSIS_PROMPT, REDUCE_HALLUCINATION_PROMPT

        available = set(headings(ENGLISH))
        for template in (FRAMING_ANALYSIS_PROMPT, REDUCE_HALLUCINATION_PROMPT):
            required = {p for p in _re.findall(r"\{(h_\w+)\}", template)}
            assert required <= available, f"missing headings: {required - available}"
