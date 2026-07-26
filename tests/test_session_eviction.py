"""
Unit tests for conversation-session eviction.

Per-session history was always capped at max_turns, but the number of sessions
was not. `/api/chat` mints a new session id for any request that arrives without
one, so every browser refresh created an entry that nothing ever removed - an
unbounded dict on a long-running server.
"""
from datetime import datetime, timedelta

import pytest

from chatbot.conversation_memory import ConversationMemory


class FrozenMemory(ConversationMemory):
    """ConversationMemory with a controllable clock, so no test has to sleep."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.now = datetime(2026, 1, 1, 12, 0, 0)

    def _now(self):
        return self.now

    def advance(self, minutes):
        self.now += timedelta(minutes=minutes)


class TestPerSessionTruncation:
    def test_history_is_capped_at_max_turns(self):
        memory = FrozenMemory(max_turns=2)

        for i in range(10):
            memory.add_message("s1", "user", f"q{i}")
            memory.add_message("s1", "assistant", f"a{i}")

        assert len(memory.get_history("s1")) == 4  # 2 turns x 2 messages

    def test_the_most_recent_messages_are_the_ones_kept(self):
        memory = FrozenMemory(max_turns=1)

        memory.add_message("s1", "user", "old")
        memory.add_message("s1", "assistant", "old reply")
        memory.add_message("s1", "user", "new")
        memory.add_message("s1", "assistant", "new reply")

        assert [m["content"] for m in memory.get_history("s1")] == ["new", "new reply"]


class TestTtlEviction:
    def test_an_idle_session_is_evicted(self):
        memory = FrozenMemory(session_ttl_minutes=60)
        memory.add_message("stale", "user", "hello")

        memory.advance(61)
        memory.add_message("fresh", "user", "hello")

        assert "stale" not in memory.sessions
        assert "fresh" in memory.sessions

    def test_a_session_within_the_ttl_survives(self):
        memory = FrozenMemory(session_ttl_minutes=60)
        memory.add_message("keep", "user", "hello")

        memory.advance(59)
        memory.add_message("other", "user", "hello")

        assert "keep" in memory.sessions

    def test_activity_refreshes_the_ttl(self):
        """An active conversation must never be evicted mid-conversation."""
        memory = FrozenMemory(session_ttl_minutes=30)
        memory.add_message("active", "user", "turn 1")

        for _ in range(5):
            memory.advance(20)
            memory.add_message("active", "user", "another turn")

        assert "active" in memory.sessions

    def test_metadata_is_removed_with_the_session(self):
        memory = FrozenMemory(session_ttl_minutes=10)
        memory.add_message("stale", "user", "hello")

        memory.advance(11)
        memory.add_message("fresh", "user", "hello")

        assert "stale" not in memory.session_metadata

    def test_eviction_reports_how_many_it_removed(self):
        memory = FrozenMemory(session_ttl_minutes=10)
        for i in range(3):
            memory.add_message(f"s{i}", "user", "hello")

        memory.advance(11)

        assert memory.evict_stale() == 3
        assert memory.sessions == {}


class TestSessionCap:
    def test_total_sessions_never_exceed_the_cap(self):
        memory = FrozenMemory(max_sessions=5, session_ttl_minutes=10_000)

        for i in range(50):
            memory.advance(1)
            memory.add_message(f"s{i}", "user", "hello")

        assert len(memory.sessions) <= 5

    def test_the_least_recently_used_session_is_dropped_first(self):
        memory = FrozenMemory(max_sessions=2, session_ttl_minutes=10_000)

        memory.add_message("oldest", "user", "hello")
        memory.advance(1)
        memory.add_message("middle", "user", "hello")
        memory.advance(1)
        memory.add_message("newest", "user", "hello")

        assert "oldest" not in memory.sessions
        assert {"middle", "newest"} <= set(memory.sessions)

    def test_a_session_being_written_to_is_never_the_one_evicted(self):
        memory = FrozenMemory(max_sessions=1, session_ttl_minutes=10_000)

        memory.add_message("first", "user", "hello")
        memory.advance(1)
        memory.add_message("second", "user", "hello")

        assert "second" in memory.sessions

    def test_create_session_also_enforces_the_cap(self):
        memory = FrozenMemory(max_sessions=3, session_ttl_minutes=10_000)

        for _ in range(20):
            memory.advance(1)
            memory.create_session()

        assert len(memory.sessions) <= 3


class TestRobustness:
    def test_a_corrupt_timestamp_is_treated_as_expired_not_crashed(self):
        memory = FrozenMemory(session_ttl_minutes=60)
        memory.add_message("broken", "user", "hello")
        memory.session_metadata["broken"]["last_activity"] = "not-a-timestamp"

        memory.evict_stale()  # must not raise

        assert "broken" not in memory.sessions

    def test_missing_metadata_is_tolerated(self):
        memory = FrozenMemory()
        memory.sessions["orphan"] = []

        memory.evict_stale()

        assert "orphan" not in memory.sessions

    def test_defaults_are_bounded(self):
        """A default build must not be unbounded."""
        memory = ConversationMemory()
        assert memory.max_sessions > 0
        assert memory.session_ttl.total_seconds() > 0


class TestStats:
    def test_stats_report_the_live_counts(self):
        memory = FrozenMemory(max_sessions=42, session_ttl_minutes=7)
        memory.add_message("s1", "user", "hello")

        assert memory.stats() == {
            "active_sessions": 1,
            "max_sessions": 42,
            "session_ttl_minutes": 7,
        }
