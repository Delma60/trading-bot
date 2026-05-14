"""
manager/episodic_memory.py — ARIA's Episodic Memory

Stores and retrieves meaningful past interactions across sessions.
This is what makes the bot feel like it knows you.
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List
from dataclasses import dataclass, asdict


@dataclass
class Episode:
    """A meaningful event worth remembering across sessions."""
    timestamp: str
    episode_type: str       # "trade", "conversation", "insight", "mistake"
    symbol: Optional[str]
    summary: str            # plain English description
    outcome: Optional[str]  # what happened as a result
    emotional_tag: Optional[str]  # "user was stressed", "big win", "recovery"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class EpisodicMemory:
    """
    Stores and retrieves meaningful past interactions.
    This is what makes the bot feel like it *knows* you.
    
    Examples of what gets stored:
    - "User took a loss on EURUSD and was frustrated. 
       They asked not to trade it for a few days."
    - "User's best trading day was when they followed 
       the XAUUSD breakout signal without hesitation."
    - "User tends to overtrade after 2 losses in a row."
    """

    MEMORY_FILE = Path("data/episodic_memory.json")
    MAX_EPISODES = 500
    BATCH_SAVE_INTERVAL = 10  # Flush writes every 10 episodes instead of every store()

    def __init__(self):
        self._episodes: List[Episode] = self._load()
        self._unsaved_count = 0

    def store(self, episode: Episode):
        self._episodes.append(episode)
        self._unsaved_count += 1

        # Prune first, then decide whether to flush
        if len(self._episodes) > self.MAX_EPISODES:
            self._episodes = self._episodes[-self.MAX_EPISODES:]

        # Flush on trades (high importance) or batch threshold
        if episode.episode_type == "trade" or self._unsaved_count >= self.BATCH_SAVE_INTERVAL:
            self.flush()

    def flush(self):
        """Explicitly flush all unsaved episodes to disk."""
        if self._unsaved_count > 0:
            self._save()
            self._unsaved_count = 0

    def recall_relevant(self, context: dict, limit: int = 3) -> List[Episode]:
        """
        Retrieves episodes relevant to the current context.
        Used to inject 'memory' into responses naturally.
        """
        symbol = context.get("symbol")
        intent = context.get("intent")
        
        scored = []
        for ep in reversed(self._episodes):
            score = 0
            if symbol and ep.symbol == symbol:
                score += 3
            if intent and intent in ep.tags:
                score += 2
            if ep.episode_type == "mistake":
                score += 1  # mistakes are worth surfacing
            if score > 0:
                scored.append((score, ep))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def recall_today(self) -> List[Episode]:
        """Get all episodes from today."""
        today = date.today().isoformat()
        return [e for e in self._episodes if e.timestamp.startswith(today)]

    def recall_pattern(self, pattern_type: str) -> Optional[str]:
        """
        Detects behavioral patterns from episode history.
        Returns a plain English description if a pattern is found.
        """
        recent = self._episodes[-50:]
        
        if pattern_type == "overtrading":
            losses = [e for e in recent 
                      if e.episode_type == "trade" and "loss" in (e.outcome or "").lower()]
            if len(losses) >= 3:
                symbols = [e.symbol for e in losses[-3:] if e.symbol]
                return f"You've taken {len(losses)} losses recently. Last few on: {', '.join(symbols)}."
        
        if pattern_type == "hesitation":
            missed = [e for e in recent if "missed" in (e.summary or "").lower()]
            if len(missed) >= 2:
                return "You've mentioned missing a few setups lately."
        
        return None

    def recall_by_symbol(self, symbol: str, limit: int = 5) -> List[Episode]:
        """Get all episodes related to a specific symbol."""
        relevant = [e for e in self._episodes if e.symbol == symbol]
        return relevant[-limit:]

    def _load(self) -> List[Episode]:
        """Load episodes from disk."""
        if not self.MEMORY_FILE.exists():
            return []
        try:
            raw = json.loads(self.MEMORY_FILE.read_text())
            episodes = []
            for r in raw:
                ep = Episode(
                    timestamp=r.get("timestamp", ""),
                    episode_type=r.get("episode_type", ""),
                    symbol=r.get("symbol"),
                    summary=r.get("summary", ""),
                    outcome=r.get("outcome"),
                    emotional_tag=r.get("emotional_tag"),
                    tags=r.get("tags", [])
                )
                episodes.append(ep)
            return episodes
        except Exception as e:
            print(f"Error loading episodic memory: {e}")
            return []

    def _save(self):
        """Save episodes to disk."""
        self.MEMORY_FILE.parent.mkdir(exist_ok=True)
        data = [asdict(e) for e in self._episodes]
        self.MEMORY_FILE.write_text(json.dumps(data, indent=2))
