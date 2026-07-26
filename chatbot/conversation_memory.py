"""
Conversation Memory Module
Manages conversation history for multi-turn chatbot interactions.
Enables contextual responses based on previous exchanges.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid

# Returned by format_for_llm() when a session has no prior turns.
# llm_client compares against this exact string, so keep them in sync.
NO_HISTORY_SENTINEL = "(No previous conversation)"

# Defaults for eviction. Sessions are created by the server on any request that
# arrives without a session_id, so every browser refresh mints a new one and
# nothing the client does ever removes it.
DEFAULT_MAX_SESSIONS = 500
DEFAULT_SESSION_TTL_MINUTES = 120


class ConversationMemory:
    """
    Stores and manages conversation history per session.

    Features:
    - Session-based storage (each user/browser gets unique session)
    - Automatic history truncation to prevent token overflow
    - Bounded session count with TTL + least-recently-used eviction
    - Formatted output for LLM context injection

    Per-session history was always capped, but the number of sessions was not:
    `self.sessions` only ever grew. A server left running accumulates one entry
    per visitor forever, holding up to 20 messages each. That is a slow leak with
    no upper bound, so sessions now expire and the total is capped.
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        session_ttl_minutes: int = DEFAULT_SESSION_TTL_MINUTES,
    ):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of conversation turns to keep per session
            max_sessions: Hard cap on concurrently tracked sessions; the least
                recently used are dropped first
            session_ttl_minutes: Sessions idle for longer than this are discarded
        """
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.session_metadata: Dict[str, Dict] = {}

    def _now(self) -> datetime:
        """Overridable clock, so eviction can be tested without sleeping."""
        return datetime.now()

    def _touch(self, session_id: str) -> None:
        """Create metadata if absent and mark the session as just used."""
        now = self._now().isoformat()
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {"created_at": now, "last_activity": now}
        else:
            self.session_metadata[session_id]["last_activity"] = now

    def _last_activity(self, session_id: str) -> datetime:
        """Parsed last-activity timestamp, treated as expired if unreadable."""
        raw = self.session_metadata.get(session_id, {}).get("last_activity")
        try:
            return datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return datetime.min

    def evict_stale(self) -> int:
        """
        Drop expired sessions, then the least recently used ones over the cap.

        Returns the number of sessions removed. Called on every write, which is
        cheap: it is a scan of at most `max_sessions` timestamps.
        """
        cutoff = self._now() - self.session_ttl
        expired = [sid for sid in self.sessions if self._last_activity(sid) < cutoff]
        for session_id in expired:
            self.delete_session(session_id)

        removed = len(expired)

        if len(self.sessions) > self.max_sessions:
            # Oldest activity first; drop until back within the cap.
            by_age = sorted(self.sessions, key=self._last_activity)
            for session_id in by_age[: len(self.sessions) - self.max_sessions]:
                self.delete_session(session_id)
                removed += 1

        if removed:
            print(f"[INFO] Evicted {removed} idle conversation session(s); "
                  f"{len(self.sessions)} active")
        return removed

    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self._touch(session_id)
        self.evict_stale()
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            session_id: The session identifier
            role: Either 'user' or 'assistant'
            content: The message content
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self._touch(session_id)

        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": self._now().isoformat()
        })

        # Truncate if exceeds max turns (1 turn = user + assistant)
        max_messages = self.max_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

        self.evict_stale()

    def stats(self) -> Dict[str, int]:
        """Session counters, for /api/health."""
        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "session_ttl_minutes": int(self.session_ttl.total_seconds() // 60),
        }
    
    def get_history(self, session_id: str) -> List[Dict]:
        """
        Get the full conversation history for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of message dictionaries with role and content
        """
        return self.sessions.get(session_id, [])
    
    def get_recent_history(self, session_id: str, num_turns: int = 5) -> List[Dict]:
        """
        Get the most recent conversation turns.
        
        Args:
            session_id: The session identifier
            num_turns: Number of recent turns to retrieve
            
        Returns:
            List of recent messages
        """
        history = self.get_history(session_id)
        max_messages = num_turns * 2
        return history[-max_messages:] if history else []
    
    def format_for_llm(self, session_id: str, num_turns: int = 5) -> str:
        """
        Format conversation history as a string for LLM context.
        Uses [USER] and [ASSISTANT] tags for parsing.
        
        Args:
            session_id: The session identifier
            num_turns: Number of recent turns to include
            
        Returns:
            Formatted string of conversation history
        """
        history = self.get_recent_history(session_id, num_turns)
        
        if not history:
            return NO_HISTORY_SENTINEL
        
        formatted_lines = []
        for msg in history:
            # Use tags that llm_client can parse
            role_tag = "[USER]" if msg["role"] == "user" else "[ASSISTANT]"
            # Truncate long messages for context efficiency
            content = msg["content"]
            if len(content) > 600:
                content = content[:600] + "..."
            formatted_lines.append(f"{role_tag} {content}")
        
        return "\n".join(formatted_lines)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear the conversation history for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session existed and was cleared, False otherwise
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Completely remove a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session existed and was deleted, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            return True
        return False
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get summary information about a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session metadata or None if not found
        """
        if session_id not in self.sessions:
            return None
        
        history = self.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": len(history),
            "turn_count": len(history) // 2,
            **self.session_metadata.get(session_id, {})
        }


# Singleton instance for application-wide use
conversation_memory = ConversationMemory()
