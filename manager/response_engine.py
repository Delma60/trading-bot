"""
manager/response_engine.py — ARIA's Response Generation Engine

Advanced response composer that creates contextually appropriate,
varied responses from reasoning engine outputs.
"""

import random
from typing import Optional


class ResponseEngine:
    """
    Advanced response composer that creates contextually appropriate,
    varied responses from reasoning engine outputs.
    """

    # ── Greetings ─────────────────────────────────────────────────────────────
    _GREETINGS = [
        "ARIA online. {account_line} {first_action}",
        "Back online. {account_line} {first_action}",
        "Ready. {account_line} {first_action}",
        "Systems live. {account_line} {first_action}",
    ]
    _NO_ACCOUNT = "Broker connected."
    _ACCOUNT_HEALTHY = "Balance ${balance:,.2f}, equity ${equity:,.2f}."
    _ACCOUNT_DRAWDOWN = "Balance ${balance:,.2f} with ${drawdown:,.2f} floating loss — watch the open positions."

    def greeting(self, account: dict, first_suggestion: str) -> str:
        balance = account.get("balance", 0)
        equity  = account.get("equity", 0)
        if balance > 0:
            drawdown = balance - equity
            if drawdown > 0:
                acct_line = self._ACCOUNT_DRAWDOWN.format(
                    balance=balance, drawdown=drawdown
                )
            else:
                acct_line = self._ACCOUNT_HEALTHY.format(
                    balance=balance, equity=equity
                )
        else:
            acct_line = self._NO_ACCOUNT

        tpl = random.choice(self._GREETINGS)
        return tpl.format(account_line=acct_line, first_action=first_suggestion)

    # ── Post-action responses ─────────────────────────────────────────────────
    _ACTION_WRAPPERS = {
        "trade_success": [
            "{action_result} {follow_up}",
            "Done — {action_result} {follow_up}",
            "{action_result} {follow_up}",
        ],
        "trade_blocked": [
            "Holding off: {action_result}",
            "Can't proceed — {action_result}",
            "Risk gate triggered: {action_result}",
        ],
        "position_closed": [
            "Closed. {action_result}",
            "{action_result} Position removed from book.",
            "Out. {action_result}",
        ],
        "data_returned": [
            "{action_result}",
            "Here's what I see: {action_result}",
            "Live data: {action_result}",
        ],
        "config_updated": [
            "Done — {action_result}",
            "Updated. {action_result}",
            "{action_result} New settings are active.",
        ],
        "generic": [
            "{action_result}",
            "Handled. {action_result}",
        ],
    }

    _FOLLOW_UPS = [
        "Want me to scan the portfolio next?",
        "Shall I check the rest of your watchlist?",
        "Monitor the position or run a wider scan?",
        "Let it breathe or set a tighter stop?",
        "Want a regime check before the next entry?",
        "Any other pairs you want analysed?",
    ]

    def action_response(
        self,
        action_result:  str,
        action_type:    str = "generic",
        add_follow_up:  bool = False,
    ) -> str:
        templates = self._ACTION_WRAPPERS.get(action_type, self._ACTION_WRAPPERS["generic"])
        tpl = random.choice(templates)
        follow_up = random.choice(self._FOLLOW_UPS) if add_follow_up else ""
        return tpl.format(action_result=action_result, follow_up=follow_up).strip()

    # ── Conversational fallbacks ───────────────────────────────────────────────
    _FALLBACKS = [
        "Not sure what you mean — scan the portfolio, check positions, or place a trade?",
        "I didn't catch that. Try: 'scan', 'positions', 'account', or name a symbol.",
        "Can you be more specific? I'm ready for a trade, scan, or account check.",
        "Didn't parse that. What do you need — analysis, a trade, or a position update?",
    ]

    _ACKNOWLEDGEMENTS = [
        "Got it. Anything else on your radar?",
        "Noted. What's next?",
        "Understood. Ready when you are.",
        "Copy that. What's the next move?",
    ]

    _GRATITUDES = [
        "Anytime. What's the plan?",
        "Happy to help. What do you need next?",
        "Let's keep the edge sharp. Anything else?",
    ]

    def fallback(self) -> str:
        return random.choice(self._FALLBACKS)

    def acknowledgement(self) -> str:
        return random.choice(self._ACKNOWLEDGEMENTS)

    def gratitude(self) -> str:
        return random.choice(self._GRATITUDES)

    # ── Proactive suggestions ─────────────────────────────────────────────────
    _SUGGESTIONS = [
        "Portfolio scan is ready to run — want me to check all {n_symbols} symbols?",
        "You've been in {symbol} for a while — want me to check its current signal?",
        "No open trades right now — good time to scan for entries across {n_symbols} pairs.",
        "Floating P&L is ${pnl:+.2f} — want to lock in any profits?",
        "Last trade was {symbol}. Want to check correlated pairs for confirmation?",
        "Regime is '{regime}' — {n_fit} strategies are well-suited. Run a scan?",
    ]

    def proactive(self, context: dict, regime: str = "Unknown", reasoning_engine=None) -> str:
        symbols   = context.get("tracked_symbols", [])
        n_symbols = len(symbols) if symbols else 0
        symbol    = context.get("last_symbol", "")
        pnl_str   = context.get("floating_pnl", "$0").replace("$", "").replace(",", "")
        try:
            pnl = float(pnl_str)
        except ValueError:
            pnl = 0.0

        # Get insights from unsupervised learner if available
        insights = []
        if reasoning_engine and reasoning_engine.learner:
            insights = reasoning_engine.learner.generate_insights()

        # Use insights to enhance suggestions
        if insights:
            # Return a relevant insight as the proactive suggestion
            relevant = [i for i in insights if "regime" in i.lower() or "strategy" in i.lower()]
            if relevant:
                return relevant[0]

        # Fallback to template-based suggestions
        # Simplified regime fit calculation (would normally import from unsupervised_learner)
        regime_fits = {
            "Strong Trend": ["Trend_Following", "Momentum", "Breakout"],
            "Ranging / Choppy": ["Mean_Reversion", "Scalping", "Arbitrage"],
            "High Volatility Breakout": ["Breakout", "Momentum"],
            "Low Volatility Consolidation": ["Mean_Reversion", "Scalping"],
        }
        n_fit = len(regime_fits.get(regime, []))

        candidates = [
            s for s in self._SUGGESTIONS
            if ("symbol" not in s or symbol)
            and ("pnl" not in s or pnl != 0)
        ]
        if not candidates:
            candidates = self._SUGGESTIONS[:2]

        tpl = random.choice(candidates)
        try:
            return tpl.format(
                n_symbols=n_symbols, symbol=symbol or "last pair",
                pnl=pnl, regime=regime, n_fit=n_fit
            )
        except (KeyError, ValueError):
            return f"Want me to scan all {n_symbols} symbols for fresh entries?"

    # ── Session debrief ───────────────────────────────────────────────────────
    def debrief(
        self,
        n_trades:    int,
        session_pnl: float,
        winning_symbols: list,
        losing_symbols:  list,
    ) -> str:
        parts = []

        # Performance headline
        if session_pnl > 0:
            parts.append(
                f"Good session — ${session_pnl:+.2f} across {n_trades} trade(s)."
            )
        elif session_pnl < 0:
            parts.append(
                f"Session ended down ${abs(session_pnl):.2f} across {n_trades} trade(s)."
            )
        else:
            parts.append(f"Break-even session. {n_trades} trade(s) taken.")

        # Winner/loser callout
        if winning_symbols:
            parts.append(f"Winners: {', '.join(winning_symbols[:3])}.")
        if losing_symbols:
            parts.append(f"Losers: {', '.join(losing_symbols[:3])}. Review those setups.")

        # Generic forward-looking sign-off
        sign_offs = [
            "Logs saved. See you next session.",
            "Data recorded for pattern learning. Rest well.",
            "History logged. The learner will process tonight's patterns.",
        ]
        parts.append(random.choice(sign_offs))
        return " ".join(parts)

    # ── Legacy compatibility method ───────────────────────────────────────────
    def generate(self, intent: str, context: dict = None) -> str:
        """Legacy method for backward compatibility with old chat.py"""
        context = context or {}

        if intent == "greeting":
            account = context.get("account", {})
            suggestion = context.get("suggestion", "Ready for analysis.")
            return self.greeting(account, suggestion)
        elif intent == "analyze_symbol":
            # Generate a comprehensive analysis response
            analysis = context.get("analysis", {})
            if analysis:
                symbol = analysis.get("symbol", "UNKNOWN")
                narration = analysis.get("narration", "")
                regime_advice = analysis.get("regime_advice", "")
                scored = analysis.get("scored", {})
                grade = scored.get("grade", "C")

                response = f"{narration} {regime_advice}"
                if grade in ["A", "B"]:
                    response += f" Quality grade: {grade} — this setup has strong potential."
                elif grade == "C":
                    response += f" Quality grade: {grade} — proceed with caution."
                else:
                    response += f" Quality grade: {grade} — I'd recommend waiting for a better setup."

                return response
            return "Analysis complete, but missing detailed data."
        elif intent == "portfolio_status":
            return self.action_response(
                f"Portfolio valued at ${context.get('total_value', '0.00')} "
                f"with {context.get('active_positions', 0)} active positions. "
                f"Daily P&L: {context.get('daily_pnl', '+0.00')}.",
                "data_returned"
            )
        else:
            return self.fallback()