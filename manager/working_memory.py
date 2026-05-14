"""
manager/working_memory.py — ARIA's Working Memory

Tracks the live context of a single conversation session.
This is the bot's 'mind' during a conversation — not just what was said,
but what it means and how it's evolving.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import deque


@dataclass
class ConversationTurn:
    """A single exchange in the conversation."""
    role: str           # "user" or "aria"
    text: str
    intent: str
    emotion: str        # detected user emotion
    timestamp: datetime = field(default_factory=datetime.now)
    action_taken: Optional[str] = None
    outcome: Optional[str] = None


class WorkingMemory:
    """
    Holds the live context of a single conversation session.
    This is the bot's 'mind' during a conversation.
    """

    def __init__(self, max_turns: int = 20):
        self.turns: deque = deque(maxlen=max_turns)
        
        # What we're currently talking about
        self.active_topic: Optional[str] = None
        self.topic_depth: int = 0           # how deep into a topic we are
        
        # Emotional arc of the conversation
        self.user_emotion_history: deque = deque(maxlen=max_turns)
        self.session_tone: str = "neutral"  # neutral, tense, confident, frustrated
        
        # What the user wants (inferred goals, not just stated)
        self.inferred_goals: list = []
        self.stated_goals: list = []
        
        # Things ARIA has committed to
        self.open_promises: list = []  # "I'll alert you when EURUSD breaks 1.08"
        
        # Trading session narrative
        self.session_start: datetime = datetime.now()
        self.symbols_discussed: list = []
        self.trades_this_session: list = []
        self.session_pnl_narrative: str = ""
        
        # Last symbol, timeframe, signal — kept for quick access
        self.last_symbol: Optional[str] = None
        self.last_signal: Optional[dict] = None
        self.last_timeframe: str = "H1"
        self.last_intent: Optional[str] = None

    def add_turn(self, turn: ConversationTurn):
        """Add a conversational turn and update session state."""
        self.turns.append(turn)
        if turn.role == "user":
            self.user_emotion_history.append(turn.emotion)
            self._update_session_tone()
        if turn.intent:
            self.last_intent = turn.intent

    def get_conversation_summary(self) -> str:
        """
        Returns a natural language summary of what's been discussed.
        Used to inject context into response generation.
        """
        if not self.turns:
            return "Session just started."
        
        recent = list(self.turns)[-5:]
        topics = list(dict.fromkeys(
            t.intent for t in recent if t.intent and t.intent != "general"
        ))
        symbols = self.symbols_discussed[-3:] if self.symbols_discussed else []
        
        parts = []
        if symbols:
            parts.append(f"We've been looking at {', '.join(symbols)}.")
        if topics:
            parts.append(f"Topics covered: {', '.join(topics)}.")
        if self.open_promises:
            parts.append(f"Still pending: {self.open_promises[0]}.")
        if self.session_tone != "neutral":
            parts.append(f"Session tone: {self.session_tone}.")
        
        return " ".join(parts)

    def _update_session_tone(self):
        """Infer session tone from recent emotion history."""
        if len(self.user_emotion_history) < 3:
            return
        recent = self.user_emotion_history[-5:]
        neg = recent.count("frustrated") + recent.count("anxious") + recent.count("negative")
        pos = recent.count("confident") + recent.count("positive") + recent.count("excited")
        if neg >= 2:
            self.session_tone = "tense"
        elif pos >= 3:
            self.session_tone = "confident"
        else:
            self.session_tone = "neutral"

    def remember_symbol(self, symbol: str):
        """Track a symbol as discussed in this session."""
        self.last_symbol = symbol
        if symbol not in self.symbols_discussed:
            self.symbols_discussed.append(symbol)

    def make_promise(self, promise: str):
        """Record something ARIA has committed to."""
        self.open_promises.append(promise)

    def fulfill_promise(self, promise: str):
        """Mark a promise as completed."""
        self.open_promises = [p for p in self.open_promises if promise not in p]

    def get_turns_text(self, limit: int = 5) -> list:
        """Get the text of recent turns for context injection."""
        recent = list(self.turns)[-limit:]
        return [(t.role, t.text) for t in recent]
