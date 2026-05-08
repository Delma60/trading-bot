"""
manager/inner_monologue.py — ARIA's Internal Reasoning

ARIA's internal reasoning process before generating a response.
The monologue shapes what gets said, how it gets said, and what gets held back.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Thought:
    """A single thought in ARIA's internal monologue."""
    category: str   # "observation", "concern", "plan", "question", "recall"
    content: str
    confidence: float = 1.0
    surface: bool = True    # whether to surface this to the user


class InnerMonologue:
    """
    ARIA's internal reasoning process before generating a response.
    
    The monologue is generated from live context and shapes:
    - What gets said
    - How it gets said  
    - What gets held back
    - What question gets asked next
    
    This separates "I see a BUY signal on EURUSD" from
    "You've been watching EURUSD for a while — it's finally 
    showing the setup you were waiting for."
    """

    def __init__(self, working_memory, episodic_memory, user_model):
        self.wm = working_memory
        self.em = episodic_memory
        self.um = user_model
        self._thoughts: List[Thought] = []

    def think(self, intent: str, entities: dict, agent_result: dict) -> List[Thought]:
        """
        Generate internal thoughts before responding.
        Returns ordered thoughts that will influence the response.
        """
        self._thoughts = []
        
        symbol = (entities.get("symbols", [None])[0] 
                 if entities.get("symbols") else None) or self.wm.last_symbol
        
        self._observe_context(intent, symbol, agent_result)
        self._check_memory(symbol, intent)
        self._assess_user_state()
        self._form_intent(intent, agent_result)
        self._consider_what_not_to_say()
        
        return self._thoughts

    def _observe_context(self, intent: str, symbol: Optional[str], result: dict):
        """What's actually happening right now?"""
        
        # Did the topic just shift?
        if self.wm.last_intent and intent != self.wm.last_intent:
            if intent in ("execute_trade", "open_buy", "open_sell"):
                self._think("observation", 
                    "User shifted from analysis to execution — they want to act.",
                    surface=False)
        
        # Is there a signal?
        action = result.get("action", "WAIT")
        conf = result.get("confidence", 0.0)
        
        if action == "WAIT" and intent in ("analyze_symbol", "ai_analysis"):
            self._think("observation",
                f"No clear signal on {symbol}. User may be frustrated if they were hoping for one.",
                surface=False)

        if action in ("BUY", "SELL") and conf > 0.75:
            self._think("plan",
                f"Strong {action} signal at {conf:.0%}. Lead with the action, support with reasoning.",
                surface=False)
        
        # How long have we been in this session?
        from datetime import datetime
        session_minutes = (datetime.now() - self.wm.session_start).total_seconds() // 60
        if session_minutes > 90:
            self._think("concern",
                "Long session. User might be fatigued — keep responses tighter.",
                surface=False)

    def _check_memory(self, symbol: Optional[str], intent: str):
        """What do I remember that's relevant?"""
        if not symbol:
            return

        relevant = self.em.recall_relevant({"symbol": symbol, "intent": intent})
        
        for episode in relevant:
            if episode.episode_type == "mistake" and episode.symbol == symbol:
                self._think("recall",
                    f"We had a bad trade on {symbol} before: {episode.summary}",
                    surface=True,
                    confidence=0.8)
                break
            
            if "hesitation" in (episode.emotional_tag or ""):
                self._think("recall",
                    f"User hesitated last time on {symbol}. They might want more conviction this time.",
                    surface=False)

        # Check behavioral patterns
        overtrading = self.em.recall_pattern("overtrading")
        if overtrading and intent in ("execute_trade", "open_buy", "open_sell"):
            self._think("concern", overtrading, surface=True, confidence=0.9)

    def _assess_user_state(self):
        """How is the user feeling right now?"""
        tone = self.wm.session_tone
        emotion_recent = self.wm.user_emotion_history[-3:] if self.wm.user_emotion_history else []
        
        if tone == "tense":
            self._think("observation",
                "User seems stressed. Keep tone calm and certain. Avoid hedging.",
                surface=False)
        
        if "frustrated" in emotion_recent:
            self._think("concern",
                "User showed frustration recently. Acknowledge before advising.",
                surface=False)
        
        trust = self.um["trust_in_bot"]
        if trust < 0.4:
            self._think("plan",
                "Trust is low. Need to show reasoning, not just conclusions.",
                surface=False)
        elif trust > 0.8:
            self._think("plan",
                "Trust is high. Can be more direct and concise.",
                surface=False)

    def _form_intent(self, intent: str, result: dict):
        """What is the most useful thing I can say right now?"""
        
        # What follow-up question would move this forward?
        if intent == "analyze_symbol":
            action = result.get("action", "WAIT")
            if action != "WAIT":
                self._think("question",
                    f"Should I ask if they want to execute the {action}?",
                    surface=False)
            else:
                self._think("question",
                    "Should I suggest a different timeframe or pair?",
                    surface=False)
        
        # Are there open promises to fulfill?
        if self.wm.open_promises:
            self._think("recall",
                f"I promised: '{self.wm.open_promises[0]}' — check if this is relevant now.",
                surface=False)

    def _consider_what_not_to_say(self):
        """Active suppression of unhelpful content."""
        
        # Don't repeat what was just said
        if len(self.wm.turns) >= 2:
            last_aria = next(
                (t for t in reversed(list(self.wm.turns)) if t.role == "aria"), None
            )
            if last_aria and "Grade" in last_aria.text:
                self._think("observation",
                    "Already gave grade last turn. Don't repeat the grade.",
                    surface=False)
        
        # Don't over-qualify when trust is high
        if self.um["trust_in_bot"] > 0.7:
            self._think("plan",
                "Skip the 'consider' and 'might' hedging. Be direct.",
                surface=False)

    def _think(self, category: str, content: str, surface: bool = True, confidence: float = 1.0):
        """Record a thought."""
        self._thoughts.append(Thought(category, content, confidence, surface))

    def get_surfaceable(self) -> List[Thought]:
        """Thoughts appropriate to mention to the user."""
        return [t for t in self._thoughts if t.surface and t.confidence >= 0.7]

    def get_directive(self) -> Optional[Thought]:
        """The single most important directive for shaping the response."""
        plans = [t for t in self._thoughts if t.category == "plan" and not t.surface]
        return plans[0] if plans else None

    def get_concern(self) -> Optional[Thought]:
        """Any concern worth flagging before proceeding."""
        concerns = [t for t in self._thoughts if t.category == "concern" and t.surface]
        return concerns[0] if concerns else None
