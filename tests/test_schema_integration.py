"""
Integration tests against a real PostgreSQL.

The unit suite fakes the database, which means nothing verified that
database/schema.sql actually executes - and the schema is exactly where the
embedding-provenance guard lives. A migration that silently failed to apply
would leave `embedding_model` absent, ingestion would raise on every insert, and
the first sign of it would be a broken app.

These tests need a database but not a running server and not the embedding
model: chunks are inserted directly, and a stub embedder stands in where one is
required. That keeps them fast enough to run in CI.

They TRUNCATE the article tables, so they refuse to run against a database whose
name does not mark it as disposable - see REQUIRED_DB_NAME_MARKERS. Running them
without that guard emptied the development corpus on the first try, which is
exactly the mistake worth making impossible rather than remembering.

    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sentra_ci \
        pytest -m integration -k schema
"""
import json
import os

import numpy as np
import pytest
import pytest_asyncio

from config.settings import settings
from database.connection import DatabaseManager, split_sql_statements
from pipeline.ingest import ArticleIngestor
from pipeline.preprocessing import ProcessedChunk

pytestmark = pytest.mark.integration

# A database is only considered disposable if its name says so. Anything else is
# assumed to hold a real corpus that took minutes to embed.
REQUIRED_DB_NAME_MARKERS = ("test", "_ci", "ci_", "scratch")


def schema_statements():
    with open("database/schema.sql", "r", encoding="utf-8") as f:
        return split_sql_statements(f.read())


def target_database_url():
    """
    The database these tests may destroy.

    SENTRA_TEST_DATABASE_URL is honoured first so a developer can point the
    destructive suite somewhere safe without touching the DATABASE_URL their
    running app uses.
    """
    return os.environ.get("SENTRA_TEST_DATABASE_URL") or settings.DATABASE_URL


def assert_disposable(url):
    """Skip rather than truncate a database that does not look like a test one."""
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0].lower()
    if not any(marker in name for marker in REQUIRED_DB_NAME_MARKERS):
        pytest.skip(
            f"refusing to truncate database {name!r}: these tests DELETE all articles. "
            f"Name it with one of {REQUIRED_DB_NAME_MARKERS}, or set "
            f"SENTRA_TEST_DATABASE_URL to a disposable database."
        )


# pytest_asyncio.fixture, not pytest.fixture: a plain fixture hands the test the
# async generator object itself rather than awaiting it.
#
# Function-scoped on purpose. A module-scoped async fixture is created in a
# different event loop from the function-scoped tests that use it, and asyncpg
# connections are bound to the loop that opened them - the result is
# "attached to a different loop" on every test.
@pytest_asyncio.fixture
async def db():
    """A connected pool with the schema applied, skipping if no database is reachable."""
    url = target_database_url()
    assert_disposable(url)

    manager = DatabaseManager(url)
    try:
        await manager.connect()
    except Exception as exc:
        pytest.skip(f"no database available: {exc}")

    # Applied directly rather than through init_database(), which swallows any
    # error containing "already exists". A migration that failed for some other
    # reason would be hidden by that; here it fails the test.
    async with manager.acquire() as conn:
        for statement in schema_statements():
            await conn.execute(statement)

    yield manager
    await manager.disconnect()


@pytest_asyncio.fixture
async def clean(db):
    """Empty the article tables around each test."""
    await db.execute("DELETE FROM article_chunks")
    await db.execute("DELETE FROM articles")
    yield db
    await db.execute("DELETE FROM article_chunks")
    await db.execute("DELETE FROM articles")


class StubEmbedder:
    """Deterministic 4-dim vectors, so no model download is needed."""

    dimension = 4

    def generate_single(self, text):
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.rand(self.dimension)

    def generate(self, texts, show_progress=False):
        return np.array([self.generate_single(t) for t in texts])


class StubPreprocessor:
    def chunk_article(self, article):
        return [ProcessedChunk(
            chunk_text=article["content"],
            chunk_index=0,
            metadata={"title": article["title"], "media_source": article["media_source"]},
        )]


@pytest.fixture
def ingestor():
    instance = object.__new__(ArticleIngestor)
    instance.embedder = StubEmbedder()
    instance.preprocessor = StubPreprocessor()
    return instance


async def column_exists(db, table, column):
    return await db.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2)",
        table, column,
    )


class TestSchemaApplies:
    @pytest.mark.asyncio
    async def test_core_tables_exist(self, db):
        for table in ("articles", "article_chunks",
                      "hallucination_labels", "confidence_labels"):
            assert await db.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = $1)", table
            ), f"{table} was not created"

    @pytest.mark.asyncio
    async def test_provenance_columns_exist(self, db):
        """
        The whole point of the migration block. If these are missing, every
        ingestion insert fails - and init_database swallows 'already exists'
        errors, so a broken ALTER could otherwise pass unnoticed.
        """
        assert await column_exists(db, "article_chunks", "embedding_model")
        assert await column_exists(db, "article_chunks", "embedding_dim")
        assert await column_exists(db, "articles", "corpus_version")

    @pytest.mark.asyncio
    async def test_corpus_version_column_fits_a_real_version_string(self, db):
        from data.corpus import corpus_version

        length = await db.fetchval(
            "SELECT character_maximum_length FROM information_schema.columns "
            "WHERE table_name = 'articles' AND column_name = 'corpus_version'"
        )
        assert len(corpus_version()) <= length

    @pytest.mark.asyncio
    async def test_schema_can_be_applied_twice(self, db):
        """
        The schema runs on every server start, so it must be idempotent. The db
        fixture already applied it once; applying it again must not raise.
        """
        async with db.acquire() as conn:
            for statement in schema_statements():
                await conn.execute(statement)


class TestIngestionRecordsProvenance:
    ARTICLE = {
        "title": "Test article",
        "content": "Prabowo Subianto won the 2024 Indonesian presidential election.",
        "media_source": "tempo",
        "url": "https://example.test/a",
        "author": "",
        "published_date": None,
        "category": "election",
    }

    @pytest.mark.asyncio
    async def test_corpus_version_is_stored_on_the_article(self, clean, ingestor):
        await ingestor.ingest_article(self.ARTICLE, db=clean, corpus_version="ci-corpus@abc123")

        assert await clean.fetchval("SELECT corpus_version FROM articles") == "ci-corpus@abc123"

    @pytest.mark.asyncio
    async def test_embedding_model_and_dim_are_stored_on_the_chunk(self, clean, ingestor):
        await ingestor.ingest_article(self.ARTICLE, db=clean, corpus_version="ci")

        row = await clean.fetchrow("SELECT embedding_model, embedding_dim, embedding FROM article_chunks")
        assert row["embedding_model"] == settings.EMBEDDING_MODEL
        assert row["embedding_dim"] == StubEmbedder.dimension
        assert len(row["embedding"]) == StubEmbedder.dimension

    @pytest.mark.asyncio
    async def test_unsupported_media_source_is_rejected(self, clean, ingestor):
        with pytest.raises(ValueError, match="Unsupported media source"):
            await ingestor.ingest_article(dict(self.ARTICLE, media_source="not_a_source"), db=clean)

    @pytest.mark.asyncio
    async def test_versions_are_reported_back(self, clean, ingestor):
        await ingestor.ingest_article(self.ARTICLE, db=clean, corpus_version="v1")
        await ingestor.ingest_article(dict(self.ARTICLE, url="https://example.test/b"),
                                      db=clean, corpus_version="v2")

        assert await ingestor.get_corpus_versions(db=clean) == {"v1": 1, "v2": 1}

    @pytest.mark.asyncio
    async def test_deleting_an_article_deletes_its_chunks(self, clean, ingestor):
        article_id = await ingestor.ingest_article(self.ARTICLE, db=clean, corpus_version="ci")

        await clean.execute("DELETE FROM articles WHERE id = $1", article_id)

        assert await clean.fetchval("SELECT COUNT(*) FROM article_chunks") == 0


class TestVerificationAgainstRealRows:
    """
    The guard's SQL runs against a real table here, not a fake. The fake cannot
    catch a typo in a column name or a GROUP BY that PostgreSQL rejects.
    """

    async def insert_chunk(self, db, model, text="a chunk", vector=None):
        article_id = await db.fetchval(
            "INSERT INTO articles (title, content, media_source) "
            "VALUES ('t', 'c', 'tempo') RETURNING id"
        )
        await db.execute(
            "INSERT INTO article_chunks "
            "(article_id, chunk_index, chunk_text, embedding, embedding_model, embedding_dim, metadata) "
            "VALUES ($1, 0, $2, $3, $4, $5, $6)",
            article_id, text,
            list(vector if vector is not None else StubEmbedder().generate_single(text)),
            model, StubEmbedder.dimension, json.dumps({}),
        )

    @pytest.mark.asyncio
    async def test_empty_database_passes(self, clean, ingestor):
        result = await ingestor.verify_embeddings_match_model(db=clean)

        assert result["ok"] is True
        assert result["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_matching_model_passes(self, clean, ingestor):
        await self.insert_chunk(clean, settings.EMBEDDING_MODEL)

        result = await ingestor.verify_embeddings_match_model(db=clean)

        assert result["ok"] is True
        assert result["total_chunks"] == 1
        assert result["min_self_similarity"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_wrong_model_label_is_detected(self, clean, ingestor):
        """The exact regression that went unnoticed for weeks."""
        await self.insert_chunk(clean, "sentence-transformers/all-MiniLM-L6-v2")

        result = await ingestor.verify_embeddings_match_model(db=clean)

        assert result["ok"] is False
        assert "all-MiniLM-L6-v2" in result["detail"]

    @pytest.mark.asyncio
    async def test_null_model_label_is_detected(self, clean, ingestor):
        """Rows written before the column existed are untrustworthy by definition."""
        await self.insert_chunk(clean, None)

        result = await ingestor.verify_embeddings_match_model(db=clean)

        assert result["ok"] is False
        assert "<unlabelled>" in result["mismatched_models"]

    @pytest.mark.asyncio
    async def test_correct_label_but_wrong_vector_is_detected(self, clean, ingestor):
        await self.insert_chunk(clean, settings.EMBEDDING_MODEL, vector=[9.0, -9.0, 9.0, -9.0])

        result = await ingestor.verify_embeddings_match_model(db=clean)

        assert result["ok"] is False
        assert result["min_self_similarity"] < 0.99
