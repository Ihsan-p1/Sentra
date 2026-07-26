"""
Unit tests for the LLM client.

No network: the HTTP call is monkeypatched. What is under test is the wiring -
endpoint detection, header construction, history parsing and context budgeting -
which is exactly the part that broke during the OpenRouter -> Ollama migration.
"""
import pytest

from chatbot import llm_client as llm_module
from chatbot.llm_client import LLMClient, _is_local_endpoint
from config.settings import settings


@pytest.fixture
def local_client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "llama3.1")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return LLMClient()


class TestEndpointDetection:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434/v1",
            "http://127.0.0.1:1234/v1",
            "http://0.0.0.0:8080/v1",
            "http://host.docker.internal:11434/v1",
        ],
    )
    def test_local_urls_are_detected(self, url):
        assert _is_local_endpoint(url) is True

    @pytest.mark.parametrize(
        "url",
        ["https://openrouter.ai/api/v1", "https://api.groq.com/openai/v1", "https://example.com/v1"],
    )
    def test_remote_urls_are_not_local(self, url):
        assert _is_local_endpoint(url) is False


class TestApiKeyHandling:
    def test_local_endpoint_needs_no_configured_key(self, local_client):
        # Regression: the client used to hard-require an OpenRouter key and
        # refuse to answer at all when pointed at a local Ollama server.
        assert local_client.is_local is True
        assert local_client.api_key  # placeholder, not empty

    def test_remote_endpoint_without_key_is_reported(self, monkeypatch):
        monkeypatch.setattr(settings, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
        monkeypatch.setattr(settings, "GROQ_API_KEY", None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        client = LLMClient()

        assert client.is_local is False
        assert client.api_key is None

    def test_base_url_trailing_slash_is_stripped(self, monkeypatch):
        monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434/v1/")
        assert LLMClient().base_url == "http://localhost:11434/v1"


class TestHeaders:
    def test_local_request_omits_openrouter_headers(self, local_client):
        headers = local_client._build_headers()

        assert headers["Authorization"].startswith("Bearer ")
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers

    def test_openrouter_request_includes_attribution_headers(self, monkeypatch):
        monkeypatch.setattr(settings, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key")

        headers = LLMClient()._build_headers()

        assert headers["HTTP-Referer"] == "http://localhost:8000"
        assert headers["X-Title"] == "Sentra News Analysis"


class TestNewsContextFormatting:
    def test_no_chunks_returns_placeholder(self, local_client):
        assert "No news sources" in local_client._format_news_context({})

    def test_context_is_capped_per_source(self, local_client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_CONTEXT_CHUNKS_PER_SOURCE", 2)
        chunks = [{"text": f"body {i}", "title": f"Title {i}"} for i in range(5)]

        context = local_client._format_news_context({"tempo": chunks})

        assert "Title 0" in context and "Title 1" in context
        assert "Title 2" not in context

    def test_chunk_text_is_truncated(self, local_client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS_PER_CHUNK", 10)

        context = local_client._format_news_context(
            {"tempo": [{"text": "x" * 500, "title": "T"}]}
        )

        assert "x" * 10 in context
        assert "x" * 11 not in context

    def test_source_name_is_uppercased(self, local_client):
        context = local_client._format_news_context({"abc_news": [{"text": "t", "title": "T"}]})
        assert "ABC_NEWS" in context

    def test_available_sources_are_declared_as_a_closed_list(self, local_client):
        # Regression: qwen2.5:3b rendered ANTARANEWS as "ANTV" until the allowed
        # names were stated explicitly instead of implied by the headings.
        context = local_client._format_news_context(
            {
                "antaranews": [{"text": "a", "title": "A"}],
                "tempo": [{"text": "t", "title": "T"}],
            }
        )

        header = context.splitlines()[0]
        assert "AVAILABLE SOURCES" in header
        assert "ANTARANEWS" in header and "TEMPO" in header

    def test_sources_without_chunks_are_not_advertised(self, local_client):
        context = local_client._format_news_context(
            {"tempo": [{"text": "t", "title": "T"}], "abc_news": []}
        )

        assert "ABC_NEWS" not in context.splitlines()[0]

    def test_all_sources_empty_returns_placeholder(self, local_client):
        assert "No relevant news" in local_client._format_news_context({"tempo": [], "abc_news": []})


class TestHistoryParsing:
    def test_tagged_history_becomes_message_dicts(self, local_client):
        history = "[USER] who won the election?\n[ASSISTANT] Prabowo Subianto won the election."

        messages = local_client._parse_history_to_messages(history)

        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "who won the election?"

    def test_multiline_messages_are_joined(self, local_client):
        history = "[ASSISTANT] first line of the answer\nsecond line of the answer"

        messages = local_client._parse_history_to_messages(history)

        assert len(messages) == 1
        assert "first line of the answer second line of the answer" == messages[0]["content"]

    def test_very_short_messages_are_dropped(self, local_client):
        messages = local_client._parse_history_to_messages("[USER] ok\n[ASSISTANT] sure thing")
        assert all(len(m["content"]) > 10 for m in messages)

    def test_history_is_limited_to_last_six_messages(self, local_client):
        history = "\n".join(f"[USER] message number {i} here" for i in range(20))

        messages = local_client._parse_history_to_messages(history)

        assert len(messages) == 6
        assert "19" in messages[-1]["content"]

    def test_untagged_text_yields_no_messages(self, local_client):
        assert local_client._parse_history_to_messages("(No previous conversation)") == []


class TestRequestPayload:
    @pytest.mark.asyncio
    async def test_strict_mode_is_deterministic_and_seeded(self, local_client, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "answer"}}]}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

        monkeypatch.setattr(llm_module.requests, "post", fake_post)

        answer = await local_client.generate_conversational_response(
            user_query="who won?",
            retrieved_chunks={"tempo": [{"text": "Prabowo won.", "title": "T"}]},
            mode="reduce_hallucination",
        )

        assert answer == "answer"
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert captured["payload"]["temperature"] == 0.0
        assert captured["payload"]["seed"] == settings.LLM_SEED
        assert captured["payload"]["model"] == "llama3.1"

    @pytest.mark.asyncio
    async def test_prompt_instructs_the_requested_response_language(self, local_client, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "jawaban"}}]}

        monkeypatch.setattr(
            llm_module.requests,
            "post",
            lambda url, json=None, headers=None, timeout=None: captured.update(payload=json) or FakeResponse(),
        )

        await local_client.generate_conversational_response(
            user_query="siapa presiden indonesia",
            retrieved_chunks={"tempo": [{"text": "Prabowo won.", "title": "T"}]},
            language="id",
        )

        system_prompt = captured["payload"]["messages"][0]["content"]
        assert "Indonesian (Bahasa Indonesia)" in system_prompt
        assert "English" not in system_prompt.split("RESPONSE LANGUAGE:")[1].split("\n")[0]

    @pytest.mark.asyncio
    async def test_english_is_the_default_response_language(self, local_client, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "answer"}}]}

        monkeypatch.setattr(
            llm_module.requests,
            "post",
            lambda url, json=None, headers=None, timeout=None: captured.update(payload=json) or FakeResponse(),
        )

        await local_client.generate_conversational_response(
            user_query="who is the president",
            retrieved_chunks={"tempo": [{"text": "Prabowo won.", "title": "T"}]},
        )

        assert "RESPONSE LANGUAGE: English" in captured["payload"]["messages"][0]["content"]

class TestFailuresRaiseInsteadOfAnswering:
    """
    Transport failures must raise LLMUnavailable, not return their message as the
    answer.

    Returning the text meant "Connection Error: could not reach
    http://localhost:11434/v1" was split into sentences, scored by the
    hallucination detector, given a confidence percentage, rendered with a full
    A/B panel, and written into conversation memory - where it became the context
    the next follow-up resolved against.
    """

    @pytest.mark.asyncio
    async def test_connection_error_raises(self, local_client, monkeypatch):
        def fake_post(*args, **kwargs):
            raise llm_module.requests.exceptions.ConnectionError()

        monkeypatch.setattr(llm_module.requests, "post", fake_post)

        with pytest.raises(llm_module.LLMUnavailable) as exc:
            await local_client.generate_conversational_response("q", {})

        assert "Connection Error" in exc.value.message
        assert "ollama serve" in exc.value.hint

    @pytest.mark.asyncio
    async def test_http_error_raises(self, local_client, monkeypatch):
        class FakeResponse:
            status_code = 404
            text = "model not found"

        monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeResponse())

        with pytest.raises(llm_module.LLMUnavailable) as exc:
            await local_client.generate_conversational_response("q", {})

        assert "404" in exc.value.message
        assert "model not found" in exc.value.hint

    @pytest.mark.asyncio
    async def test_timeout_raises_with_a_remedy(self, local_client, monkeypatch):
        def fake_post(*args, **kwargs):
            raise llm_module.requests.exceptions.Timeout()

        monkeypatch.setattr(llm_module.requests, "post", fake_post)

        with pytest.raises(llm_module.LLMUnavailable) as exc:
            await local_client.generate_conversational_response("q", {})

        assert "LLM_TIMEOUT" in exc.value.hint

    @pytest.mark.asyncio
    async def test_no_choices_raises(self, local_client, monkeypatch):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": []}

        monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeResponse())

        with pytest.raises(llm_module.LLMUnavailable, match="empty response"):
            await local_client.generate_conversational_response("q", {})

    @pytest.mark.asyncio
    async def test_blank_content_raises(self, local_client, monkeypatch):
        """An all-whitespace completion is not an answer either."""
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "   \n  "}}]}

        monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeResponse())

        with pytest.raises(llm_module.LLMUnavailable, match="empty message"):
            await local_client.generate_conversational_response("q", {})

    def test_user_message_joins_message_and_hint(self):
        exc = llm_module.LLMUnavailable("Timeout Error.", "Raise LLM_TIMEOUT.")
        assert exc.user_message() == "Timeout Error. Raise LLM_TIMEOUT."

    def test_user_message_without_a_hint_has_no_trailing_space(self):
        assert llm_module.LLMUnavailable("Broken.").user_message() == "Broken."
