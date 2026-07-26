"""
Unit tests for the JSON corpus and its loader.

The corpus used to be a Python literal in data/election_articles.py, which meant
it could not be validated without executing it. Now that it is data, these tests
check both the loader and the corpus content itself - including the specific gap
that made "who won the 2024 election?" unanswerable.
"""
import json

import pytest

from config.settings import settings
from data.corpus import (
    CORPUS_PATH,
    REQUIRED_FIELDS,
    corpus_metadata,
    corpus_version,
    load_articles,
)


@pytest.fixture(scope="module")
def articles():
    return load_articles()


@pytest.fixture(scope="module")
def raw():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestLoader:
    def test_corpus_loads(self, articles):
        assert len(articles) > 100

    def test_every_article_has_the_required_fields(self, articles):
        for article in articles:
            for field in REQUIRED_FIELDS:
                assert article.get(field), f"{field} missing from {article.get('title')!r}"

    def test_dates_are_parsed_to_datetime(self, articles):
        dated = [a for a in articles if a["published_date"] is not None]
        assert dated, "no article carries a published_date"
        for article in dated:
            assert hasattr(article["published_date"], "year")

    def test_loader_shape_matches_what_ingestion_expects(self, articles):
        # ArticleIngestor.ingest_article reads exactly these keys
        expected = {"title", "content", "media_source", "url", "author",
                    "published_date", "category"}
        assert set(articles[0]) == expected

    def test_missing_required_fields_are_skipped_not_crashed(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({
            "name": "broken",
            "articles": [
                {"title": "ok", "content": "body", "media_source": "tempo"},
                {"title": "no content", "media_source": "tempo"},
            ],
        }), encoding="utf-8")

        loaded = load_articles(str(broken))

        assert len(loaded) == 1
        assert loaded[0]["title"] == "ok"

    def test_unparseable_date_becomes_none_rather_than_raising(self, tmp_path):
        odd = tmp_path / "odd.json"
        odd.write_text(json.dumps({
            "name": "odd",
            "articles": [{
                "title": "t", "content": "c", "media_source": "tempo",
                "published_date": "not-a-date",
            }],
        }), encoding="utf-8")

        assert load_articles(str(odd))[0]["published_date"] is None


class TestCorpusVersion:
    def test_version_is_stable_across_calls(self):
        assert corpus_version() == corpus_version()

    def test_version_names_the_corpus_and_carries_a_digest(self):
        name, _, digest = corpus_version().partition("@")
        assert name
        assert len(digest) == 12

    def test_version_fits_the_database_column(self):
        # articles.corpus_version is VARCHAR(64)
        assert len(corpus_version()) <= 64

    def test_editing_an_article_changes_the_version(self, tmp_path, raw):
        def write(name, articles):
            path = tmp_path / name
            path.write_text(json.dumps({"name": "c", "articles": articles}), encoding="utf-8")
            return str(path)

        one = raw["articles"][:3]
        two = [dict(one[0], title="changed")] + one[1:]

        assert corpus_version(write("a.json", one)) != corpus_version(write("b.json", two))

    def test_changing_only_the_description_does_not_change_the_version(self, tmp_path):
        article = {"title": "t", "content": "c", "media_source": "tempo"}

        def write(name, description):
            path = tmp_path / name
            path.write_text(
                json.dumps({"name": "c", "description": description, "articles": [article]}),
                encoding="utf-8",
            )
            return str(path)

        assert corpus_version(write("a.json", "one")) == corpus_version(write("b.json", "two"))


class TestCorpusContent:
    def test_every_source_is_supported(self, articles):
        for article in articles:
            assert article["media_source"] in settings.SUPPORTED_MEDIA

    def test_all_supported_sources_are_present(self, articles):
        present = {a["media_source"] for a in articles}
        assert present == set(settings.SUPPORTED_MEDIA)

    def test_metadata_counts_match_the_loaded_articles(self, articles):
        meta = corpus_metadata()
        assert meta["article_count"] == len(articles)
        assert sum(meta["counts_by_source"].values()) == len(articles)

    def test_no_duplicate_urls(self, articles):
        """
        Two URLs were duplicated: the same source article had been ingested twice
        under two different paraphrases. That inflates the article count, gives
        retrieval two near-identical chunks to return, and double-counts one
        outlet's vocabulary in the framing comparison.
        """
        urls = [a["url"] for a in articles if a["url"]]
        duplicates = {u for u in urls if urls.count(u) > 1}
        assert not duplicates, f"duplicate URLs: {duplicates}"

    def test_no_typographic_punctuation(self, articles):
        """
        Curly apostrophes and non-breaking hyphens change tokenisation:
        "Indonesia's" and "quick-count" become different tokens from their ASCII
        spellings, which silently splits TF-IDF term counts and weakens the
        keyword-overlap baseline. Accented letters in names are fine.
        """
        offenders = {
            "‘": "left single quote", "’": "right single quote",
            "“": "left double quote", "”": "right double quote",
            "‑": "non-breaking hyphen", "–": "en dash", "—": "em dash",
        }

        found = {
            name for article in articles
            for char, name in offenders.items()
            if char in article["title"] + article["content"]
        }

        assert not found, f"normalise these to ASCII: {sorted(found)}"

    def test_the_election_winner_is_stated_explicitly(self, articles):
        """
        The corpus originally named the 2024 winner exactly once, in passing,
        inside a quick-count article. That single mention ranked too low to be
        retrieved, so the most basic question a user can ask was unanswerable for
        reasons no threshold or prompt change could fix. Articles stating the
        official result were added; this test stops them being dropped again.
        """
        winner_articles = [
            a for a in articles
            if "prabowo" in a["content"].lower()
            and any(w in a["content"].lower() for w in ("won", "winner", "victor"))
        ]

        assert len(winner_articles) >= 3, (
            "too few articles state the election winner - "
            "'who won the 2024 election?' will be refused"
        )

    def test_the_winner_is_named_in_the_opening_sentence_of_some_article(self, articles):
        """
        Chunking is positional: a fact buried at the end of a long article lands
        in a later chunk and competes with everything else in the corpus. At
        least one article must lead with the result.
        """
        leads = [
            a for a in articles
            if "prabowo" in a["content"][:200].lower()
            and any(w in a["content"][:200].lower() for w in ("won", "winner", "victor", "elected"))
        ]

        assert leads, "no article opens by naming the election winner"

    def test_the_official_vote_count_is_present(self, articles):
        """A verifiable number makes the answer checkable rather than merely fluent."""
        assert any("96,214,691" in a["content"] for a in articles)
        assert any("58.59" in a["content"] for a in articles)
