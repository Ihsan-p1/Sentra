"""
Unit tests for embedding-model and corpus-version provenance.

An embedding is only comparable to another embedding from the same model, and
that fact used to be untracked. EMBEDDING_MODEL was changed from an English-only
model to a multilingual one without re-ingesting; both emit 384 dimensions, so
nothing failed and retrieval silently compared vectors across two embedding
spaces for weeks.

These tests exercise the guard with a fake database rather than a real one, so
they run offline.
"""
import numpy as np
import pytest

from config.settings import settings
from pipeline.ingest import ArticleIngestor


class FakeDB:
    """Answers only the two queries verify_embeddings_match_model issues."""

    def __init__(self, label_counts, chunks=None):
        self.label_counts = label_counts
        self.chunks = chunks or []

    async def fetch(self, query, *args):
        if "GROUP BY embedding_model" in query:
            return [{"embedding_model": m, "n": n} for m, n in self.label_counts.items()]
        if "chunk_text, embedding" in query:
            limit = args[0] if args else len(self.chunks)
            return self.chunks[:limit]
        if "GROUP BY corpus_version" in query:
            return [{"corpus_version": "test-corpus@abc123", "n": 3}]
        raise AssertionError(f"unexpected query: {query}")


class StubEmbedder:
    """Returns a fixed vector per text, so self-similarity is exactly 1.0."""

    dimension = 4

    def __init__(self, drifted=False):
        self.drifted = drifted

    def _vector(self, text):
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.rand(self.dimension)

    def generate_single(self, text):
        vector = self._vector(text)
        if self.drifted:
            # Simulate a model whose weights changed behind an unchanged name
            vector = vector[::-1] + 0.5
        return vector

    def generate(self, texts, show_progress=False):
        return np.array([self._vector(t) for t in texts])


@pytest.fixture
def ingestor():
    """An ArticleIngestor with a stub embedder and no preprocessor wiring."""
    instance = object.__new__(ArticleIngestor)
    instance.embedder = StubEmbedder()
    return instance


def chunk_rows(embedder, texts):
    return [{"chunk_text": t, "embedding": list(embedder.generate_single(t))} for t in texts]


class TestModelLabelCheck:
    @pytest.mark.asyncio
    async def test_empty_database_is_not_a_failure(self, ingestor):
        result = await ingestor.verify_embeddings_match_model(db=FakeDB({}))

        assert result["ok"] is True
        assert result["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_matching_labels_and_vectors_pass(self, ingestor):
        texts = ["chunk one", "chunk two"]
        db = FakeDB({settings.EMBEDDING_MODEL: 2}, chunk_rows(ingestor.embedder, texts))

        result = await ingestor.verify_embeddings_match_model(db=db)

        assert result["ok"] is True
        assert result["min_self_similarity"] == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_a_different_model_label_fails_immediately(self, ingestor):
        db = FakeDB({"sentence-transformers/all-MiniLM-L6-v2": 275})

        result = await ingestor.verify_embeddings_match_model(db=db)

        assert result["ok"] is False
        assert "all-MiniLM-L6-v2" in result["detail"]
        assert "refresh_database" in result["detail"]
        # Caught by label alone - no re-embedding needed
        assert result["checked"] == 0

    @pytest.mark.asyncio
    async def test_unlabelled_chunks_fail(self, ingestor):
        """Rows predating the column are exactly the untrustworthy case."""
        result = await ingestor.verify_embeddings_match_model(db=FakeDB({None: 275}))

        assert result["ok"] is False
        assert "<unlabelled>" in result["mismatched_models"]

    @pytest.mark.asyncio
    async def test_partially_mismatched_table_fails(self, ingestor):
        db = FakeDB({settings.EMBEDDING_MODEL: 200, "some/other-model": 75})

        result = await ingestor.verify_embeddings_match_model(db=db)

        assert result["ok"] is False
        assert result["mismatched_models"] == {"some/other-model": 75}

    @pytest.mark.asyncio
    async def test_the_expected_model_is_reported(self, ingestor):
        db = FakeDB({settings.EMBEDDING_MODEL: 1}, chunk_rows(ingestor.embedder, ["a chunk"]))

        assert (await ingestor.verify_embeddings_match_model(db=db))["model"] == settings.EMBEDDING_MODEL


class TestVectorCheck:
    @pytest.mark.asyncio
    async def test_right_label_but_drifted_vectors_still_fails(self):
        """
        The label check cannot catch this: the name is correct while the weights
        behind it are not. That is what the re-embedding sample is for.
        """
        instance = object.__new__(ArticleIngestor)
        stored = StubEmbedder()
        rows = chunk_rows(stored, ["chunk one", "chunk two"])
        instance.embedder = StubEmbedder(drifted=True)

        result = await instance.verify_embeddings_match_model(
            db=FakeDB({settings.EMBEDDING_MODEL: 2}, rows)
        )

        assert result["ok"] is False
        assert result["min_self_similarity"] < 0.99
        assert "may have changed" in result["detail"]

    @pytest.mark.asyncio
    async def test_dimension_mismatch_is_reported_clearly(self, ingestor):
        rows = [{"chunk_text": "a chunk", "embedding": [0.1] * 384}]
        db = FakeDB({settings.EMBEDDING_MODEL: 1}, rows)

        result = await ingestor.verify_embeddings_match_model(db=db)

        assert result["ok"] is False
        assert "dimension mismatch" in result["detail"]
        assert "384" in result["detail"]


class TestCorpusVersions:
    @pytest.mark.asyncio
    async def test_versions_are_reported(self, ingestor):
        versions = await ingestor.get_corpus_versions(db=FakeDB({}))

        assert versions == {"test-corpus@abc123": 3}
