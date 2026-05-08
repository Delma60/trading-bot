"""
manager/voice_layer.py — ARIA's Voice/Naturalizer

Transforms structured agent output + cognitive context into
natural, varied, personality-consistent responses.
"""

import random
from typing import Optional, List
from datetime import datetime


class VoiceLayer:
    """
    Takes structured agent output + cognitive context and produces
    natural, varied, personality-consistent responses.
    
    This is not a template picker. It assembles responses from
    semantic building blocks, shaped by:
    - What was just said (avoid repetition)
    - User's emotional state
    - Communication preferences
    - Inner monologue directives
    - What's been said this session
    """

    # ARIA's personality constants
    PERSONA = {
        "name": "ARIA",
        "voice": "direct, intelligent, calm under pressure",
        "never": ["amazing", "certainly", "absolutely", "of course", "great question"],
        "contractions": True,
        "first_person": True,
    }

    def __init__(self, working_memory, user_model, inner_monologue):
        self.wm = working_memory
        self.um = user_model
        self.il = inner_monologue
        self.last_followthrough_question = None  # Track last question to avoid repetition

    def render(self, agent_output: str, thoughts: List, intent: str) -> str:
        """
        Transform raw agent output into natural speech.
        """
        style = self.um.get_communication_style()
        directive = self.il.get_directive()
        concern = self.il.get_concern()
        recalls = [t for t in thoughts if t.category == "recall" and t.surface]
        
        parts = []

        # 1. Surface any memory recall first — it shows you remember
        if recalls and random.random() > 0.4:  # don't always do it, feels natural
            parts.append(self._render_recall(recalls[0]))

        # 2. Surface any concern before the main content
        if concern:
            parts.append(self._render_concern(concern.content))

        # 3. Core content — adapted by directive
        core = self._adapt_core(agent_output, directive, style)
        parts.append(core)

        # 4. Natural follow-through (not every time)
        followthrough = self._maybe_followthrough(intent, agent_output, style)
        if followthrough:
            parts.append(followthrough)

        response = " ".join(p for p in parts if p)
        response = self._apply_personality(response)
        response = self._vary_sentence_starts(response)
        
        return response

    def _adapt_core(self, content: str, directive, style: dict) -> str:
        """Shape core content based on directive."""
        if not directive:
            return content
        
        d = directive.content.lower()
        
        if "direct" in d or "concise" in d:
            # Strip hedging language
            content = content.replace("You might want to consider ", "")
            content = content.replace("It seems like ", "")
            content = content.replace("potentially ", "")
        
        if "show reasoning" in d or "trust is low" in d:
            # The agent output already has reasoning — make sure it leads
            pass
        
        if "calm" in d or "stressed" in d:
            # Remove exclamation marks, urgency language
            content = content.replace("!", ".")
            content = content.replace("immediately", "when ready")
        
        return content

    def _render_recall(self, thought) -> str:
        """Turn a memory into a natural conversational reference."""
        content = thought.content
        
        openers = [
            "Worth noting — ",
            "Keeping in mind that ",
            "I remember that ",
            "Just so you know — ",
        ]
        
        return random.choice(openers) + content.lower().rstrip(".") + "."

    def _render_concern(self, concern: str) -> str:
        """Surface a concern in a non-alarming way."""
        openers = [
            "One thing first — ",
            "Before we go further, ",
            "Just flagging — ",
        ]
        return random.choice(openers) + concern

    def _maybe_followthrough(self, intent: str, content: str, style: dict) -> Optional[str]:
        """
        Generate a natural follow-up question or suggestion.
        Only sometimes — not after every single response.
        Avoids repeating the same question twice in a row.
        """
        if random.random() > 0.45:  # ~45% of the time
            return None
        
        action_taken = any(w in content for w in ["executed", "closed", "opened"])
        has_signal = any(w in content for w in ["BUY", "SELL", "Grade A", "Grade B"])
        
        if action_taken:
            options = [
                "Want me to set an alert when it hits target?",
                "Shall I keep watching it?",
                "Want to scan the rest of the portfolio while we wait?",
            ]
        elif has_signal and intent == "analyze_symbol":
            options = [
                "Want to take it?",
                "Ready to pull the trigger?",
                "Should I size the position?",
            ]
        elif intent == "account_summary":
            options = [
                "Want a scan to find something to do with it?",
                "Check open positions?",
            ]
        else:
            return None
        
        # Exclude last question to avoid repetition
        available = [q for q in options if q != self.last_followthrough_question]
        if not available:
            available = options
        
        chosen = random.choice(available)
        self.last_followthrough_question = chosen
        return chosen

    def _apply_personality(self, text: str) -> str:
        """Remove words ARIA never says."""
        for word in self.PERSONA["never"]:
            text = text.replace(f"{word} ", "")
            text = text.replace(f"{word}!", ".")
        return text

    def _vary_sentence_starts(self, text: str) -> str:
        """
        Detect if the same sentence opener was used in the last turn.
        If so, restructure it.
        """
        last_turns = list(self.wm.turns)
        if not last_turns:
            return text
        
        last_aria = next(
            (t.text for t in reversed(last_turns) if t.role == "aria"), ""
        )
        
        first_word = text.split()[0] if text.split() else ""
        last_first_word = last_aria.split()[0] if last_aria.split() else ""
        
        # If we're starting the same way as last time, restructure
        if first_word and first_word == last_first_word:
            if text.startswith("No "):
                text = text[3:].capitalize() + " — nothing there."
            elif text.startswith("The "):
                text = text[4:].capitalize()
        
        return text

    def render_greeting(self, agent_output: str, user_ctx: dict) -> str:
        """
        Greeting is special — it should reference history naturally.
        """
        sessions = user_ctx.get("sessions", 0)
        last_seen = user_ctx.get("last_seen")
        
        if sessions == 1:
            prefix = ""
        elif sessions < 5:
            prefix = "Back again. "
        elif last_seen:
            try:
                last_dt = datetime.fromisoformat(last_seen)
                days_ago = (datetime.now() - last_dt).days
                if days_ago == 0:
                    prefix = ""
                elif days_ago == 1:
                    prefix = "Morning. "
                elif days_ago <= 3:
                    prefix = f"Been {days_ago} days. "
                else:
                    prefix = f"Welcome back. {days_ago} days since we last traded. "
            except Exception:
                prefix = ""
        else:
            prefix = ""
        
        return prefix + agent_output
