"""
Conversation Memory Module
Manages conversation history for multi-turn chatbot interactions.
Enables contextual responses based on previous exchanges.
"""
from typing import Dict, List, Optional
from datetime import datetime
import uuid


class ConversationMemory:
    """
    Stores and manages conversation history per session.
    
    Features:
    - Session-based storage (each user/browser gets unique session)
    - Automatic history truncation to prevent token overflow
    - Formatted output for LLM context injection
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep per session
        """
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_turns = max_turns
        self.session_metadata: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self.session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
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
            self.session_metadata[session_id] = {
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update last activity
        self.session_metadata[session_id]["last_activity"] = datetime.now().isoformat()
        
        # Truncate if exceeds max turns (1 turn = user + assistant)
        max_messages = self.max_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]
    
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
            return "(No previous conversation)"
        
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
