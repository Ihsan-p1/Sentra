"""Tests for building the full-text corpus from saved pages (synthetic fixtures, no outlet text)."""
import shutil
from pathlib import Path

from scripts.fulltext.build_corpus import build_corpus
from scripts.fulltext.fetch_antara import raw_path

FIXTURES = Path(__file__).parent / "fixtures" / "fulltext"
ANTARA = "https://en.antaranews.com/news/304425/lantern-festival"
TEMPO = "https://en.tempo.co/read/1791076/harbour-ferry-fares"
TYPOGRAPHIC = "‘’“”‑–—"


def entry(url, outlet, genre="news", event="e1", title="Manifest title"):
    return {"url": url, "outlet": outlet, "event": event, "genre": genre, "title": title,
            "published_date": "2024-01-01", "date_source": "page", "found_by": "test"}


def save(raw_dir, url, outlet, fixture):
    path = raw_path(url, outlet, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / fixture, path)
    return path


def test_build_reports_missing_raw_file(tmp_path):
    news, opinion, errors = build_corpus([entry(ANTARA, "antaranews")], tmp_path)

    expected = raw_path(ANTARA, "antaranews", tmp_path)
    assert len(errors) == 1
    assert ANTARA in errors[0] and str(expected) in errors[0]
    assert news["articles"] == [] and opinion["articles"] == []


def test_build_reports_empty_body(tmp_path):
    # A page saved in another format, or before it finished loading, has no article body.
    path = raw_path(TEMPO, "tempo", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("<html><body><p>Loading...</p></body></html>", encoding="utf-8")

    news, _, errors = build_corpus([entry(TEMPO, "tempo")], tmp_path)

    assert len(errors) == 1
    assert "empty body" in errors[0] and str(path) in errors[0]
    assert news["articles"] == []


def test_opinion_goes_to_separate_payload(tmp_path):
    save(tmp_path, ANTARA, "antaranews", "antara_article.html")
    save(tmp_path, TEMPO, "tempo", "tempo_article.html")
    entries = [entry(ANTARA, "antaranews"), entry(TEMPO, "tempo", genre="opinion")]

    news, opinion, errors = build_corpus(entries, tmp_path)

    assert errors == []
    assert [a["url"] for a in news["articles"]] == [ANTARA]
    assert [a["url"] for a in opinion["articles"]] == [TEMPO]
    assert opinion["articles"][0]["genre"] == "opinion"


def test_payload_keys_and_sources(tmp_path):
    save(tmp_path, ANTARA, "antaranews", "antara_article.html")

    news, opinion, errors = build_corpus([entry(ANTARA, "antaranews", event="e3")], tmp_path)

    assert errors == []
    for payload in (news, opinion):
        assert {"name", "description", "sources", "no_train_sources", "unevaluated_models",
                "articles"} <= set(payload)
        assert payload["sources"] == ["antaranews", "tempo"]
        assert payload["no_train_sources"] == ["tempo"]
        assert payload["unevaluated_models"] == ["framing_bert"]
    assert news["name"] == "indonesia-election-2024-fulltext"
    assert news["name"] != opinion["name"]

    article = news["articles"][0]
    assert set(article) == {"title", "content", "media_source", "url", "author", "published_date",
                            "category", "event", "genre"}
    assert article["title"] == "Manifest title"
    assert article["media_source"] == "antaranews"
    assert article["event"] == "e3" and article["genre"] == "news"
    # The page date wins over the manifest (11 of 26 Tempo id-estimates were a day early).
    assert article["published_date"] == "2024-02-14"
    assert article["author"] == "Dewi Example, Budi Sample"
    assert article["content"].startswith("Thousands of visitors gathered along the river")
    assert len(article["content"].split("\n\n")) == 4


def test_content_is_ascii(tmp_path):
    # The Tempo fixture carries curly double quotes in its body.
    save(tmp_path, TEMPO, "tempo", "tempo_article.html")

    news, _, errors = build_corpus([entry(TEMPO, "tempo", title="Fares ‘flat’")], tmp_path)

    assert errors == []
    article = news["articles"][0]
    assert not any(c in article["title"] + article["content"] for c in TYPOGRAPHIC)
    assert article["title"] == "Fares 'flat'"
