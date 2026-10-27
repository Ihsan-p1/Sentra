"""
Sentence splitting used by the hallucination check.

The evaluation scripts call the same functions as the engine, so these tests
pin the exact rules: which sentences are checked and what text is checked.
"""
from chatbot.sentences import prepare_for_check, split_sentences


def test_split_on_punctuation_and_newlines():
    text = "## Direct Answer\nPrabowo won. Gibran is VP!\n\n- Turnout rose?"
    assert split_sentences(text) == [
        "## Direct Answer",
        "Prabowo won.",
        "Gibran is VP!",
        "- Turnout rose?",
    ]


def test_split_ignores_blank_text():
    assert split_sentences("  \n\n ") == []


def test_heading_short_and_refusal_sentences_are_skipped():
    assert prepare_for_check("## Framing Analysis of the coverage") is None
    assert prepare_for_check("## Key Points From Sources") is None
    assert prepare_for_check("Prabowo won it.") is None  # 3 words
    assert prepare_for_check("The retrieved sources do not contain that detail.") is None
    assert prepare_for_check("There is not sufficient information on this topic.") is None
    assert prepare_for_check("No explicit framing is visible in these reports.") is None


def test_citation_and_bullet_are_removed():
    assert (
        prepare_for_check("- Prabowo won 58.59% of the vote [Tempo].")
        == "Prabowo won 58.59% of the vote ."
    )


def test_citation_only_sentence_is_skipped():
    assert prepare_for_check("[Antara] [Tempo] [ABC] [Kompas]") is None


def test_sentence_left_with_two_words_after_cleaning_is_skipped():
    assert prepare_for_check("- Prabowo won [Tempo] [Antara]") is None
