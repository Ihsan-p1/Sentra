"""
Unit tests for the query-embedding seam.

`expand_query` is currently the identity function. A "News about: " prefix lived
here and was removed: it measured as a large improvement, but only because the
stored embeddings came from a different model than the query embeddings at the
time. After re-ingesting, it gave identical error rates and slightly worse
separation. These tests keep the seam honest and testable.
"""
import pytest

from rag.retrieval import QUERY_PREFIX, RAGRetriever, expand_query


class TestExpandQuery:
    def test_prefix_is_applied(self):
        assert expand_query("prabowo cabinet").startswith(QUERY_PREFIX)

    def test_original_query_is_preserved(self):
        assert "prabowo cabinet appointments" in expand_query("prabowo cabinet appointments")

    def test_surrounding_whitespace_is_trimmed(self):
        assert expand_query("  prabowo  ") == f"{QUERY_PREFIX}prabowo"

    def test_indonesian_queries_pass_through(self):
        expanded = expand_query("siapa itu gibran rakabuming")
        assert expanded.startswith(QUERY_PREFIX)
        assert "gibran" in expanded

    def test_empty_query_returns_just_the_prefix(self):
        assert expand_query("") == QUERY_PREFIX

    def test_expansion_is_deterministic(self):
        assert expand_query("election results") == expand_query("election results")

    def test_no_prefix_is_configured(self):
        # Guard: reinstating a prefix must be a deliberate, re-measured decision,
        # not something that creeps back in.
        assert QUERY_PREFIX == ""


class TestRetrieverUsesTheSeam:
    """The retriever must embed via expand_query, not the raw string."""

    class FakeEmbedder:
        def __init__(self):
            self.seen = []

        def generate_single(self, text):
            import numpy as np

            self.seen.append(text)
            return np.zeros(4)

    class FakeDB:
        async def fetch(self, query, *args):
            return []

    @pytest.mark.asyncio
    async def test_embedding_is_computed_from_the_expanded_query(self):
        embedder = self.FakeEmbedder()
        retriever = RAGRetriever(db=self.FakeDB(), embedder=embedder)

        await retriever.retrieve("prabowo cabinet appointments")

        assert embedder.seen, "embedder was never called"
        assert embedder.seen[0] == expand_query("prabowo cabinet appointments")

    @pytest.mark.asyncio
    async def test_no_sources_yields_empty_results_per_media(self):
        retriever = RAGRetriever(db=self.FakeDB(), embedder=self.FakeEmbedder())

        results = await retriever.retrieve("anything")

        assert all(chunks == [] for chunks in results.values())
