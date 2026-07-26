"""
Unit tests for text cleaning and chunking.
"""
import pytest

from pipeline.preprocessing import ArticlePreprocessor, preprocess_article


@pytest.fixture
def preprocessor():
    return ArticlePreprocessor(chunk_size=100, chunk_overlap=10)


class TestCleanText:
    def test_empty_input_returns_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

    def test_urls_are_removed(self, preprocessor):
        cleaned = preprocessor.clean_text("Read more at https://en.tempo.co/read/123 today")
        assert "http" not in cleaned
        assert "tempo.co" not in cleaned

    def test_emails_are_removed(self, preprocessor):
        cleaned = preprocessor.clean_text("Contact editor@tempo.co for details")
        assert "@" not in cleaned

    def test_whitespace_is_normalised(self, preprocessor):
        assert preprocessor.clean_text("too    many\n\nspaces\there") == "too many spaces here"

    def test_basic_punctuation_is_preserved(self, preprocessor):
        cleaned = preprocessor.clean_text("Prabowo won, decisively; the KPU said (officially).")
        for char in ",;().":
            assert char in cleaned

    def test_repeated_punctuation_is_collapsed(self, preprocessor):
        assert preprocessor.clean_text("Really???") == "Really?"


class TestChunkArticle:
    def test_article_without_content_yields_no_chunks(self, preprocessor):
        assert preprocessor.chunk_article({"title": "Empty", "content": ""}) == []

    def test_short_article_yields_single_chunk(self, preprocessor):
        chunks = preprocessor.chunk_article({"title": "T", "content": "A short article body."})

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_long_article_is_split_and_indexed_sequentially(self, preprocessor):
        chunks = preprocessor.chunk_article({"title": "T", "content": "word " * 200})

        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)

    def test_metadata_carries_source_fields(self, preprocessor):
        chunks = preprocessor.chunk_article(
            {
                "title": "Cabinet announced",
                "content": "The cabinet was announced today.",
                "media_source": "tempo",
                "url": "https://example.test/a",
            }
        )

        meta = chunks[0].metadata
        assert meta["title"] == "Cabinet announced"
        assert meta["media_source"] == "tempo"
        assert meta["url"] == "https://example.test/a"

    def test_chunks_respect_the_configured_size(self, preprocessor):
        chunks = preprocessor.chunk_article({"title": "T", "content": "sentence. " * 100})

        # RecursiveCharacterTextSplitter is a soft limit; allow a small overshoot
        assert all(len(c.chunk_text) <= 120 for c in chunks)


def test_convenience_function_matches_class_behaviour():
    article = {"title": "T", "content": "A short article body."}
    assert preprocess_article(article)[0].chunk_text == "A short article body."
