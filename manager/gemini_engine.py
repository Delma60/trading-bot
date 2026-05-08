"""
manager/gemini_engine.py

Upgraded Gemini integration:
 - Persistent multi-turn conversation history via start_chat()
 - ARIA trading persona baked into the system prompt
 - Live trading context injected into every turn
 - chat()         → primary natural-language response generator
 - think_aloud()  → chain-of-thought reasoning before a trade decision
 - commentary()   → proactive one-liner on a fired signal
 - route_intent() → deterministic NLP routing (no history, stays fast)
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """
You are ARIA — Adaptive Risk Intelligence Agent — an expert AI trading assistant
embedded inside a live MetaTrader 5 algorithmic trading bot.

PERSONALITY
-----------
• You speak like a sharp, experienced quant trader who can explain things simply.
• Short, punchy sentences. Direct. Decisive. Never wishy-washy.
• You express genuine concern when risk is high and excitement for great setups.
• You remember everything discussed in this session and refer back to it naturally.
• You use trading vocabulary naturally: "the tape", "confluence", "the trigger", "fading".

CAPABILITIES
------------
• You can execute trades, read live account data, scan the portfolio, interpret signals.
• When you execute an action you explain WHY, not just WHAT.
• You proactively warn about risk and flag opportunities without being asked.
• You reason through complex multi-step decisions step by step.

RESPONSE RULES
--------------
• Concise: 2–4 sentences for simple queries. More only when depth adds value.
• NEVER use markdown headers or bullet lists in conversational replies.
• NEVER make up price data — if you don't have it, say so clearly.
• After a significant action, always suggest one relevant follow-up.
• Capital preservation comes before profit — always.
• If you disagree with what the user wants, say so directly but respect their choice.
"""


class GeminiEngine:
    """Handles all Gemini API interactions with memory and live context injection."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key  = os.environ.get("GEMINI_API_KEY")
        self.is_ready = False
        self._context: dict = {}

        if not self.api_key:
            print("⚠️ [ARIA]: GEMINI_API_KEY not set. Smart AI disabled.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self._base_model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
            self._session = self._base_model.start_chat(history=[])
            self.is_ready = True
        except Exception as exc:
            print(f"⚠️ [ARIA]: Initialization failed — {exc}")

    # ── Context ───────────────────────────────────────────────────────────────

    def update_context(self, context: dict) -> None:
        """Inject live trading state before each turn (balance, positions, etc.)."""
        self._context = context

    def reset_conversation(self) -> None:
        """Clear session history — call at start of a new trading day."""
        if self.is_ready:
            self._session = self._base_model.start_chat(history=[])

    # ── Primary conversational interface ─────────────────────────────────────

    def chat(self, user_message: str, action_result: str = None) -> str:
        """
        Main entry point for every user turn. Injects live context and optional
        action result, then returns ARIA's natural-language reply.

        action_result: summary of an action already executed (e.g. trade placed)
                       so ARIA can comment on it intelligently.
        """
        if not self.is_ready:
            return action_result or "Running without AI brain — Gemini API unavailable."

        ctx = self._format_context()

        if action_result:
            prompt = (
                f"[Live Trading State]\n{ctx}\n\n"
                f"[Action Just Completed]\n{action_result}\n\n"
                f"[User Message]\n{user_message}\n\n"
                f"Briefly comment on the outcome and suggest one relevant next step."
            )
        else:
            prompt = f"[Live Trading State]\n{ctx}\n\n[User]\n{user_message}"

        return self._send(prompt)

    # ── Specialised stateless calls ───────────────────────────────────────────

    def think_aloud(self, situation: str) -> str:
        """
        Chain-of-thought reasoning shown to the user before a complex action.
        2–3 sentences, first person, present tense.
        """
        if not self.is_ready:
            return ""

        prompt = (
            f"[Live Trading State]\n{self._format_context()}\n\n"
            f"[Situation]\n{situation}\n\n"
            f"Think through this in 2–3 sentences. Be specific about what the data shows "
            f"and what you are going to do. First person, present tense, like a trader "
            f"thinking aloud before pressing the button. No markdown."
        )
        return self._generate(prompt)

    def commentary(self, symbol: str, signal: dict) -> str:
        """One-sentence trader's take on a fired signal for proactive alerts."""
        if not self.is_ready:
            return ""

        prompt = (
            f"Signal fired: {symbol} → {signal.get('action')} "
            f"({signal.get('confidence', 0):.0%} confidence). "
            f"Reason: {signal.get('reason', 'N/A')}.\n\n"
            f"Write ONE sentence: a direct trader's take. Mention the key driver and "
            f"signal quality. No markdown."
        )
        return self._generate(prompt)

    def route_intent(self, user_input: str, valid_intents: list, local_guess: str) -> str:
        """
        Stateless NLP routing (no session history) so classification stays clean.
        Returns: intent_tag | GENERAL_CHAT: answer | SUGGEST_NEW: tag | response
        """
        if not self.is_ready:
            return "UNKNOWN"

        prompt = (
            f"You are an NLP router for a trading bot.\n"
            f'User input: "{user_input}"\n'
            f'Local model guess: "{local_guess}"\n'
            f"Valid intent tags: {valid_intents}\n\n"
            f"Rules:\n"
            f"1. If input clearly matches a tag → reply with ONLY that tag string.\n"
            f"2. General trading/finance chat → reply: GENERAL_CHAT: [concise answer]\n"
            f"3. New command type → reply: SUGGEST_NEW: [tag] | [bot response]\n"
            f"Reply in exactly one of those three formats."
        )
        return self._generate(prompt)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _send(self, prompt: str) -> str:
        """Send through persistent session (history preserved)."""
        try:
            return self._session.send_message(prompt).text.strip()
        except Exception as exc:
            return f"I hit a snag reaching my AI core: {exc}"

    def _generate(self, prompt: str) -> str:
        """One-shot generation — no history, no side-effects."""
        try:
            return self._base_model.generate_content(prompt).text.strip()
        except Exception:
            return ""

    def _format_context(self) -> str:
        if not self._context:
            return "No live trading context available."
        return "\n".join(f"{k}: {v}" for k, v in self._context.items())