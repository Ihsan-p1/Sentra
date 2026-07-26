"""
Unit tests for ConversationMemory.

Pure in-memory state machine - no database, no LLM, no network.
"""
import pytest

from chatbot.conversation_memory import NO_HISTORY_SENTINEL, ConversationMemory


@pytest.fixture
def memory():
    return ConversationMemory(max_turns=3)


def test_unknown_session_has_empty_history(memory):
    assert memory.get_history("nope") == []


def test_add_message_creates_session_implicitly(memory):
    memory.add_message("s1", "user", "who won the election?")

    history = memory.get_history("s1")
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "who won the election?"


def test_history_preserves_order(memory):
    memory.add_message("s1", "user", "first question")
    memory.add_message("s1", "assistant", "first answer")
    memory.add_message("s1", "user", "second question")

    assert [m["role"] for m in memory.get_history("s1")] == ["user", "assistant", "user"]


def test_history_is_truncated_to_max_turns(memory):
    # max_turns=3 means at most 6 stored messages
    for i in range(10):
        memory.add_message("s1", "user", f"question {i}")
        memory.add_message("s1", "assistant", f"answer {i}")

    history = memory.get_history("s1")
    assert len(history) == 6
    # Oldest turns dropped, newest kept
    assert history[-1]["content"] == "answer 9"
    assert all("question 0" != m["content"] for m in history)


def test_sessions_are_isolated(memory):
    memory.add_message("s1", "user", "belongs to s1")
    memory.add_message("s2", "user", "belongs to s2")

    assert len(memory.get_history("s1")) == 1
    assert memory.get_history("s2")[0]["content"] == "belongs to s2"


def test_format_for_llm_returns_sentinel_when_empty(memory):
    assert memory.format_for_llm("fresh-session") == NO_HISTORY_SENTINEL


def test_format_for_llm_uses_parseable_tags(memory):
    memory.add_message("s1", "user", "who is the president?")
    memory.add_message("s1", "assistant", "Prabowo Subianto, according to the sources.")

    formatted = memory.format_for_llm("s1")

    assert "[USER] who is the president?" in formatted
    assert "[ASSISTANT] Prabowo Subianto, according to the sources." in formatted


def test_format_for_llm_truncates_long_messages(memory):
    memory.add_message("s1", "assistant", "x" * 900)

    formatted = memory.format_for_llm("s1")

    assert formatted.endswith("...")
    assert len(formatted) < 900


def test_get_recent_history_limits_turns(memory):
    for i in range(5):
        memory.add_message("s1", "user", f"q{i}")
        memory.add_message("s1", "assistant", f"a{i}")

    recent = memory.get_recent_history("s1", num_turns=1)

    assert len(recent) == 2
    assert recent[-1]["content"] == "a4"


def test_clear_session_empties_but_keeps_session(memory):
    memory.add_message("s1", "user", "hello")

    assert memory.clear_session("s1") is True
    assert memory.get_history("s1") == []
    assert memory.clear_session("does-not-exist") is False


def test_delete_session_removes_metadata(memory):
    memory.add_message("s1", "user", "hello")

    assert memory.delete_session("s1") is True
    assert memory.get_session_summary("s1") is None
    assert memory.delete_session("s1") is False


def test_session_summary_counts_turns(memory):
    memory.add_message("s1", "user", "q")
    memory.add_message("s1", "assistant", "a")

    summary = memory.get_session_summary("s1")

    assert summary["message_count"] == 2
    assert summary["turn_count"] == 1
