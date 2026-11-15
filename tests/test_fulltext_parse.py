"""Tests for the full-text article parser and cleaner (synthetic fixtures, no outlet text)."""
from pathlib import Path

import pytest

from pipeline.fulltext.parse import clean_paragraphs, parse_article, to_ascii

FIXTURES = Path(__file__).parent / "fixtures" / "fulltext"


def read_fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_antara_fixture():
    article = parse_article(read_fixture("antara_article.html"), "antaranews")
    paragraphs = clean_paragraphs(article["paragraphs"], "antaranews")

    assert article["title"] == "Lantern festival draws crowds to riverside town"
    assert article["published_date"] == "2024-02-14"
    assert article["author"] == "Dewi Example, Budi Sample"
    assert paragraphs == [
        "Thousands of visitors gathered along the river on Wednesday evening to watch "
        "the annual lantern festival, organisers said.",
        "The festival committee printed four thousand programme leaflets this year.",
        '"We started folding the lanterns in early January," the committee head told reporters.',
        "Local vendors reported that sales of grilled corn doubled compared with last year.",
    ]
    joined = " ".join(article["paragraphs"])
    for leaked in ("Fishermen welcome calm seas", "Boat race returns", "News Politics",
                   "Paper lanterns float", "festival committee printed four thousand programme "
                   "leaflets this year. The festival committee"):
        assert leaked not in joined


def test_parse_tempo_fixture():
    article = parse_article(read_fixture("tempo_article.html"), "tempo")
    paragraphs = clean_paragraphs(article["paragraphs"], "tempo")

    assert article["title"] == "Harbour Ferry Fares to Stay Flat Through Holiday Season"
    assert article["published_date"] == "2024-03-21"
    assert article["author"] == "Dewi Example"
    assert paragraphs == [
        "The harbour authority said on Thursday that ferry fares would not rise during the "
        "holiday season.",
        "Officials expect about twelve thousand passengers a day at the main pier.",
        "“We have added two evening crossings,” a harbour spokesperson told reporters.",
        "Ticket counters will open an hour earlier than usual.",
    ]
    joined = " ".join(article["paragraphs"])
    for leaked in ("Night Market", "night market", "Port Workers", "News Opinion",
                   "ferry leaves the harbour", "Translator", "March 21, 2024 |"):
        assert leaked not in joined


@pytest.mark.parametrize("line", [
    "ANTARA | DEWI EXAMPLE",
    "ANTARA",
    "Dewi Example",
    "JOHANES SAMPLE | DANIEL A. EXAMPLE",
])
def test_drops_tempo_trailing_credit(line):
    body = ["The vote count continued into the night.", line,
            "Editor’s Choice: Another headline",
            "Click here to get the latest news updates from Tempo in Google News"]
    assert clean_paragraphs(body, "tempo") == ["The vote count continued into the night."]


def test_keeps_short_closing_sentence():
    body = ["The vote count continued into the night.", "He declined to comment."]
    assert clean_paragraphs(body, "tempo") == body


@pytest.mark.parametrize("line, outlet", [
    ("Jakarta (ANTARA) - The court", "antaranews"),
    ("Surabaya, E Java (ANTARA) - The court", "antaranews"),
    ("Jakarta (ANTARA) – The court", "antaranews"),
    ("TEMPO.CO, Jakarta - The court", "tempo"),
    ("TEMPO.CO, Jakarta- The court", "tempo"),
])
def test_dateline_variants(line, outlet):
    assert clean_paragraphs([line], outlet) == ["The court"]


@pytest.mark.parametrize("line", [
    "Click here to get the latest news updates from Tempo on Google News",
    "Click here to get the latest news updates from Tempo in Google News",
    "Editor's Choice: Another headline",
    "Read the Complete Story in Tempo English Magazine",
    "Read also: Something else happened",
    "Editor: Citra Placeholder",
    "Translator: Dewi Example",
    "Reporter Ilona Estherina January 29, 2026 03:06 pm",
])
def test_drops_boilerplate_lines(line):
    body = "The vote count continued into the night."
    assert clean_paragraphs([body, line], "tempo") == [body]
    assert clean_paragraphs([body, line], "antaranews") == [body]


def test_dateline_only_stripped_from_first_paragraph():
    later = "Jakarta (ANTARA) - quoted inside a later paragraph"
    assert clean_paragraphs(["First.", later], "antaranews") == ["First.", later]


def test_to_ascii():
    text = "’ “ ” – —   ‑ ‘"
    assert to_ascii(text) == "' \" \" - -   - '"


def test_unknown_outlet_raises():
    with pytest.raises(ValueError, match="outlet"):
        parse_article("<html></html>", "abc_news")
