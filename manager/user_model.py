"""
manager/user_model.py — ARIA's User Model

A continuously updated model of the user's personality,
trading style, and preferences. Built through observation.
"""

import json
from pathlib import Path
from datetime import datetime


class UserModel:
    """
    A continuously updated model of the user's personality,
    trading style, and preferences.
    
    This is what makes ARIA feel like it knows its user —
    not because it was told, but because it observed.
    """

    MODEL_FILE = Path("data/user_model.json")

    DEFAULTS = {
        "risk_appetite":      "moderate",   # conservative / moderate / aggressive
        "trading_style":      "unknown",    # scalper / swing / position
        "experience_level":   "intermediate",
        "communication_pref": "concise",    # concise / detailed / conversational
        "stress_threshold":   3,            # losses before stress signals appear
        "decision_speed":     "deliberate", # impulsive / deliberate
        "trust_in_bot":       0.5,          # 0-1, builds over time
        "preferred_session":  "unknown",    # london / new_york / asian
        "loss_aversion":      "normal",     # low / normal / high
        "confirmation_seeker": False,       # asks bot to confirm before acting
        "follows_signals":    0.5,          # ratio of times they follow bot signals
        "overrides_bot":      0.0,          # ratio of times they override
        "total_sessions":     0,
        "last_seen":          None,
    }

    def __init__(self):
        self._model = {**self.DEFAULTS}
        self._load()

    def observe(self, event: str, context: dict = None):
        """
        Update the user model based on observed behavior.
        Called throughout the session.
        """
        context = context or {}

        if event == "followed_signal":
            n = self._model.get("follows_signals", 0.5)
            self._model["follows_signals"] = round(n * 0.9 + 1.0 * 0.1, 3)
            self._model["trust_in_bot"] = min(1.0, self._model["trust_in_bot"] + 0.02)

        elif event == "overrode_signal":
            self._model["overrides_bot"] = round(
                self._model.get("overrides_bot", 0) * 0.9 + 1.0 * 0.1, 3
            )
            self._model["trust_in_bot"] = max(0.0, self._model["trust_in_bot"] - 0.01)

        elif event == "asked_for_confirmation":
            self._model["confirmation_seeker"] = True

        elif event == "frustration_detected":
            self._model["stress_threshold"] = max(1, self._model["stress_threshold"] - 1)

        elif event == "rapid_decisions":
            self._model["decision_speed"] = "impulsive"

        elif event == "requested_detail":
            self._model["communication_pref"] = "detailed"

        elif event == "session_start":
            self._model["total_sessions"] = self._model.get("total_sessions", 0) + 1
            self._model["last_seen"] = datetime.now().isoformat()
        elif event == "asked_for_execution":
            # User requested a trade — mild trust signal
            self._model["trust_in_bot"] = min(1.0,
                self._model["trust_in_bot"] + 0.01)
        self._save()

    def get_communication_style(self) -> dict:
        """
        Returns parameters that shape how ARIA speaks to this user.
        """
        pref = self._model.get("communication_pref", "concise")
        trust = self._model.get("trust_in_bot", 0.5)
        style = self._model.get("trading_style", "unknown")

        return {
            "verbosity":       "high" if pref == "detailed" else "low",
            "add_reasoning":   trust < 0.6,   # explain more when trust is low
            "use_hedging":     trust < 0.4,   # "I think..." vs "This is..."
            "ask_confirmation": self._model.get("confirmation_seeker", False),
            "style_hint":      style,
        }

    def get_greeting_context(self) -> dict:
        """What should ARIA reference when greeting the user?"""
        return {
            "sessions": self._model.get("total_sessions", 0),
            "last_seen": self._model.get("last_seen"),
            "trust":     self._model.get("trust_in_bot", 0.5),
            "style":     self._model.get("trading_style", "unknown"),
        }

    def infer_risk_appetite(self, recent_lot_sizes: list, account_balance: float):
        """Infer risk appetite from position sizing behavior."""
        if not recent_lot_sizes or account_balance <= 0:
            return
        avg_risk = sum(recent_lot_sizes) / len(recent_lot_sizes)
        risk_pct = (avg_risk * 1000) / account_balance  # rough estimate
        if risk_pct < 0.5:
            self._model["risk_appetite"] = "conservative"
        elif risk_pct > 2.0:
            self._model["risk_appetite"] = "aggressive"
        else:
            self._model["risk_appetite"] = "moderate"
        self._save()

    def _load(self):
        """Load user model from disk."""
        if self.MODEL_FILE.exists():
            try:
                stored = json.loads(self.MODEL_FILE.read_text())
                self._model.update(stored)
            except Exception:
                pass

    def _save(self):
        """Save user model to disk."""
        self.MODEL_FILE.parent.mkdir(exist_ok=True)
        self.MODEL_FILE.write_text(json.dumps(self._model, indent=2))

    def __getitem__(self, key):
        """Dict-like access to model attributes."""
        return self._model.get(key, self.DEFAULTS.get(key))

    def __setitem__(self, key, value):
        """Dict-like setting of model attributes."""
        self._model[key] = value
        self._save()

    def get(self, key, default=None):
        """Dict-like get with default."""
        return self._model.get(key, default)
