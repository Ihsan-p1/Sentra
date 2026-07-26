"""
Unit tests for /api/health.

The endpoint used to return a literal `{"status": "ok", "models": "loaded"}`. It
reported healthy before initialisation had run, with the database unreachable,
and with Model A missing - a health check that could not fail. It is also the
first endpoint anyone reviewing this project will call.

Collaborators are faked, so no database, server, or model weights are needed.
"""
import pytest

import api.routes as routes
from config.settings import settings


class FakeMemory:
    @staticmethod
    def stats():
        return {"active_sessions": 3, "max_sessions": 500, "session_ttl_minutes": 120}


class FakeChatbot:
    def __init__(self, model_a_trained=True, bert_loaded=True):
        self.hallucination_detector = type("D", (), {"is_trained": model_a_trained})()
        self.confidence_scorer = type("S", (), {"is_trained": model_a_trained})()
        self.bert_framing = type("B", (), {"is_loaded": bert_loaded})()
        self.memory = FakeMemory()


class FakeIngestor:
    def __init__(self, versions=None, fail=False):
        self.versions = versions if versions is not None else {"corpus@abc123": 154}
        self.fail = fail

    async def get_corpus_versions(self, db=None):
        if self.fail:
            raise ConnectionError("could not connect")
        return self.versions


class FakeDb:
    def __init__(self, fail=False):
        self.fail = fail

    async def fetchval(self, query, *args):
        if self.fail:
            raise ConnectionError("could not connect")
        return 283


@pytest.fixture
def wire(monkeypatch):
    """Install fake components into the routes module."""
    def _wire(chatbot=None, ingestor=None, db=None):
        monkeypatch.setattr(routes, "chatbot", chatbot)
        monkeypatch.setattr(routes, "ingestor", ingestor)
        monkeypatch.setattr(routes, "db_manager", db if db is not None else FakeDb())
    return _wire


class TestHealthyState:
    @pytest.mark.asyncio
    async def test_fully_wired_system_reports_ok(self, wire):
        wire(FakeChatbot(), FakeIngestor())

        result = await routes.health_check()

        assert result["status"] == "ok"
        assert result["degraded"] == []

    @pytest.mark.asyncio
    async def test_corpus_details_are_reported(self, wire):
        wire(FakeChatbot(), FakeIngestor())

        corpus = (await routes.health_check())["corpus"]

        assert corpus["versions"] == {"corpus@abc123": 154}
        assert corpus["chunks"] == 283
        assert corpus["embedding_model"] == settings.EMBEDDING_MODEL
        assert corpus["consistent"] is True

    @pytest.mark.asyncio
    async def test_llm_target_is_reported(self, wire):
        wire(FakeChatbot(), FakeIngestor())

        llm = (await routes.health_check())["llm"]

        assert llm["model"] == settings.LLM_MODEL
        assert llm["base_url"] == settings.LLM_BASE_URL

    @pytest.mark.asyncio
    async def test_session_counters_are_exposed(self, wire):
        """Makes the eviction cap observable rather than a hidden constant."""
        wire(FakeChatbot(), FakeIngestor())

        assert (await routes.health_check())["checks"]["sessions"]["active_sessions"] == 3


class TestDegradedStates:
    @pytest.mark.asyncio
    async def test_uninitialised_components_are_not_ok(self, wire):
        """The exact case the old endpoint reported as healthy."""
        wire(None, None)

        result = await routes.health_check()

        assert result["status"] == "degraded"
        assert "components_initialized" in result["degraded"]

    @pytest.mark.asyncio
    async def test_untrained_model_a_is_degraded_not_ok(self, wire):
        """The engine falls back to untrained models and only warns in a log."""
        wire(FakeChatbot(model_a_trained=False), FakeIngestor())

        result = await routes.health_check()

        assert result["status"] == "degraded"
        assert "model_a_trained" in result["degraded"]

    @pytest.mark.asyncio
    async def test_missing_framing_bert_is_degraded(self, wire):
        wire(FakeChatbot(bert_loaded=False), FakeIngestor())

        assert "framing_bert_loaded" in (await routes.health_check())["degraded"]

    @pytest.mark.asyncio
    async def test_unreachable_database_is_degraded(self, wire):
        wire(FakeChatbot(), FakeIngestor(fail=True))

        result = await routes.health_check()

        assert result["status"] == "degraded"
        assert "database_reachable" in result["degraded"]
        assert "error" in result["corpus"]

    @pytest.mark.asyncio
    async def test_a_database_failure_does_not_raise(self, wire):
        """A health check that 500s tells a monitor nothing useful."""
        wire(FakeChatbot(), FakeIngestor(), FakeDb(fail=True))

        assert (await routes.health_check())["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_mixed_corpus_versions_are_degraded(self, wire):
        """A partial re-ingest leaves two corpora in one database."""
        wire(FakeChatbot(), FakeIngestor(versions={"a@1": 100, "b@2": 54}))

        result = await routes.health_check()

        assert result["corpus"]["consistent"] is False
        assert "corpus_consistent" in result["degraded"]
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_an_empty_database_is_consistent(self, wire):
        """No articles is not the same as a mixed corpus."""
        wire(FakeChatbot(), FakeIngestor(versions={}))

        assert (await routes.health_check())["corpus"]["consistent"] is True


class TestApiIngestProvenance:
    def test_api_ingested_articles_carry_a_corpus_version(self):
        """
        Without it these rows store corpus_version NULL, which the startup
        consistency check correctly flags as a mixed-version corpus - the endpoint
        would have made the app warn about itself.
        """
        assert routes.API_CORPUS_VERSION
        assert len(routes.API_CORPUS_VERSION) <= 64  # articles.corpus_version column

    def test_published_date_is_accepted(self):
        """It was absent from the model, so every API article stored a NULL date."""
        assert "published_date" in routes.ArticleRequest.model_fields
