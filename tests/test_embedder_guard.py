"""
Unit tests for the embedder singleton guard.

EmbeddingGenerator is a singleton so the weights load once. That made it ignore
its own argument: `EmbeddingGenerator("some/other-model")` returned the
already-loaded instance and served vectors from a different model than the caller
asked for, with no warning. That is the same failure that put mismatched vectors
in the database and degraded retrieval silently for weeks.

These tests install a fake loaded model on the class rather than downloading real
weights, so they run offline and fast.
"""
import numpy as np
import pytest

from config.settings import settings
from pipeline.embeddings import EmbeddingGenerator


class FakeSentenceTransformer:
    def __init__(self, dimension=384):
        self._dimension = dimension

    def get_embedding_dimension(self):
        return self._dimension

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.zeros(self._dimension)
        return np.zeros((len(texts), self._dimension))


@pytest.fixture
def loaded_singleton():
    """
    Pretend the singleton is already loaded with the configured model, then put
    the class back exactly as it was.
    """
    saved_instance = EmbeddingGenerator._instance
    saved_model = EmbeddingGenerator._model

    instance = object.__new__(EmbeddingGenerator)
    instance.model_name = settings.EMBEDDING_MODEL
    instance._model = FakeSentenceTransformer()
    instance._dimension = 384

    EmbeddingGenerator._instance = instance
    EmbeddingGenerator._model = instance._model

    yield instance

    EmbeddingGenerator._instance = saved_instance
    EmbeddingGenerator._model = saved_model


class TestModelNameGuard:
    def test_requesting_a_different_model_raises(self, loaded_singleton):
        with pytest.raises(ValueError, match="already loaded"):
            EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")

    def test_the_error_explains_why_it_matters(self, loaded_singleton):
        with pytest.raises(ValueError) as exc:
            EmbeddingGenerator("some/other-model")

        message = str(exc.value)
        assert "not comparable" in message
        assert "re-ingest" in message
        assert settings.EMBEDDING_MODEL in message

    def test_requesting_the_same_model_is_fine(self, loaded_singleton):
        again = EmbeddingGenerator(settings.EMBEDDING_MODEL)

        assert again is loaded_singleton

    def test_requesting_no_model_uses_the_configured_one(self, loaded_singleton):
        again = EmbeddingGenerator()

        assert again is loaded_singleton
        assert again.model_name == settings.EMBEDDING_MODEL

    def test_the_loaded_model_is_not_replaced_by_a_failed_request(self, loaded_singleton):
        with pytest.raises(ValueError):
            EmbeddingGenerator("some/other-model")

        assert loaded_singleton.model_name == settings.EMBEDDING_MODEL


class TestDimension:
    def test_dimension_is_resolved_once_not_per_call(self, loaded_singleton):
        """
        The property used to call get_sentence_embedding_dimension() on every
        access. Verification reads it per sampled chunk, so a deprecated-accessor
        FutureWarning was printed several times on every startup.
        """
        calls = {"n": 0}

        def counting_accessor():
            calls["n"] += 1
            return 384

        loaded_singleton._model.get_embedding_dimension = counting_accessor

        for _ in range(5):
            assert loaded_singleton.dimension == 384

        assert calls["n"] == 0, "dimension must be cached, not re-read"

    def test_the_new_accessor_name_is_preferred(self):
        """
        sentence-transformers renamed get_sentence_embedding_dimension ->
        get_embedding_dimension. Preferring the new name keeps this working when
        the old one is removed.
        """
        class BothNames:
            @staticmethod
            def get_embedding_dimension():
                return 768

            @staticmethod
            def get_sentence_embedding_dimension():
                raise AssertionError("deprecated accessor was used")

        instance = object.__new__(EmbeddingGenerator)
        instance._model = BothNames()

        assert instance._read_dimension() == 768

    def test_the_old_accessor_is_still_accepted(self):
        """Older installed versions expose only the deprecated name."""
        class OldOnly:
            @staticmethod
            def get_sentence_embedding_dimension():
                return 384

        instance = object.__new__(EmbeddingGenerator)
        instance._model = OldOnly()

        assert instance._read_dimension() == 384

    def test_dimension_is_measured_when_no_accessor_exists(self):
        """Last resort: encode something and look at the shape."""
        class NoAccessor:
            @staticmethod
            def encode(text, **kwargs):
                return np.zeros(512)

        instance = object.__new__(EmbeddingGenerator)
        instance._model = NoAccessor()

        assert instance._read_dimension() == 512
