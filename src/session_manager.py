"""
session_manager.py — Conversation Memory & Session Context for multi-turn dialogue.

Session lifecycle:
    - Sessions are keyed by userId (one per user).
    - Maximum 20 turns per session. Reset via '/reset' or user switch.
    - Stores: conversation_history, last_recommendations, last_intent, last_query.

Multi-turn use case (Dạng 4):
    TC1: "What should I watch tonight?" → system recommends Pulp Fiction
    TC4: "Why do you think I'd like that?" → system resolves "that" via last_recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

MAX_HISTORY_TURNS: int = 20
MAX_HISTORY_IN_PAYLOAD: int = 3   # Only last 3 turns sent to LLM


@dataclass
class SessionContext:
    """Stores state for one user's conversation."""
    user_id: int
    conversation_history: list[dict] = field(default_factory=list)
    last_recommendations: Optional[list[dict]] = None  # [{movieId, title, genres, reason}, ...]
    last_discussed_movie: Optional[dict] = None        # Movie analyzed in peer analysis or discussed
    last_intent: Optional[str] = None
    prev_intent: Optional[str] = None
    last_query: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to conversation history, pruning oldest if over limit."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep at most MAX_HISTORY_TURNS turns
        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]

    def get_recent_history(self, n_turns: int = MAX_HISTORY_IN_PAYLOAD) -> list[dict]:
        """Return the n most recent turns for LLM payload."""
        return self.conversation_history[-(n_turns * 2):]

    def set_recommendations(self, recs: list[dict]) -> None:
        """Store the latest recommendations (for follow-up explanation)."""
        self.last_recommendations = recs

    def reset(self) -> None:
        """Reset the session (clear all history and context)."""
        self.conversation_history.clear()
        self.last_recommendations = None
        self.last_discussed_movie = None
        self.last_intent = None
        self.prev_intent = None
        self.last_query = None

    @property
    def has_recommendations(self) -> bool:
        return bool(self.last_recommendations) or bool(self.last_discussed_movie)

    @property
    def turn_count(self) -> int:
        return len(self.conversation_history) // 2


class SessionManager:
    """Central manager for all user sessions (in-memory)."""

    def __init__(self) -> None:
        self._sessions: dict[int, SessionContext] = {}

    def get(self, user_id: int) -> SessionContext:
        """Get or create a session for the given user."""
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionContext(user_id=user_id)
        return self._sessions[user_id]

    def reset(self, user_id: int) -> None:
        """Reset a specific user's session."""
        if user_id in self._sessions:
            self._sessions[user_id].reset()

    def reset_all(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    def exists(self, user_id: int) -> bool:
        return user_id in self._sessions


# Global singleton session manager instance
session_manager = SessionManager()
