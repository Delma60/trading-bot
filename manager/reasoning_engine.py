"""
manager/reasoning_engine.py — ARIA's Internal Reasoning Core

A fully self-contained rule-based + data-driven reasoning system.
No external APIs. No LLMs. Everything computed from live trading state.

Architecture
------------
1. SignalNarrator     — translates raw strategy dicts into plain English sentences
2. TradeReasoner      — scores trade quality using multi-factor analysis
3. RiskCommentator    — builds risk-aware warnings from live account data
4. RegimeAdvisor      — maps market regime to strategy recommendations
5. ResponseComposer   — assembles scored, varied, context-filled responses

Design principle: every output is *earned* — it references real numbers
from live context, not generic filler.
"""

import random
import math
from datetime import datetime
from typing import Optional
from collections import Counter

try:
    from .unsupervised_learner import UnsupervisedLearner
    _LEARNER_AVAILABLE = True
except ImportError:
    _LEARNER_AVAILABLE = False


# ── Signal narration ──────────────────────────────────────────────────────────

class SignalNarrator:
    """Turns a strategy signal dict into a readable English sentence."""

    _BUY_OPENERS = [
        "The setup on {symbol} looks constructive —",
        "{symbol} is showing a buy signal —",
        "Bullish confluence on {symbol}:",
        "Long bias on {symbol} —",
        "{symbol} is flagging an entry —",
    ]
    _SELL_OPENERS = [
        "Bearish pressure building on {symbol} —",
        "{symbol} is flipping short —",
        "Sell signal triggered on {symbol}:",
        "Short bias on {symbol} —",
        "{symbol} showing distribution —",
    ]
    _WAIT_OPENERS = [
        "{symbol} has no clear edge right now —",
        "Staying flat on {symbol} —",
        "No conviction on {symbol} yet —",
        "{symbol} is in a neutral zone —",
    ]

    _CONF_QUALIFIERS = {
        (0.85, 1.01): ["high-conviction", "strong", "well-confirmed", "clear"],
        (0.65, 0.85): ["moderate-conviction", "reasonably confirmed", "developing"],
        (0.45, 0.65): ["tentative", "low-conviction", "early-stage"],
        (0.00, 0.45): ["weak", "marginal", "borderline"],
    }

    def narrate(self, signal: dict, symbol: str) -> str:
        action = signal.get("action", "WAIT")
        conf   = signal.get("confidence", 0.0)
        reason = signal.get("reason", "")
        source = signal.get("source", "strategy engine")

        # Pick opener
        if action == "BUY":
            opener = random.choice(self._BUY_OPENERS).format(symbol=symbol)
        elif action == "SELL":
            opener = random.choice(self._SELL_OPENERS).format(symbol=symbol)
        else:
            opener = random.choice(self._WAIT_OPENERS).format(symbol=symbol)
            return f"{opener} waiting for a cleaner setup."

        # Confidence qualifier
        qualifier = "confirmed"
        for (lo, hi), words in self._CONF_QUALIFIERS.items():
            if lo <= conf < hi:
                qualifier = random.choice(words)
                break

        # Trim reason to first clause
        short_reason = reason.split(".")[0].split("|")[0].strip()
        short_reason = short_reason[:120] if short_reason else "multiple factors aligning"

        conf_pct = f"{conf:.0%}"
        return f"{opener} {qualifier} signal at {conf_pct}. {short_reason}."


# ── Trade quality scoring ─────────────────────────────────────────────────────

class TradeReasoner:
    """
    Multi-factor trade quality scoring.

    Score components
    ----------------
    - Signal confidence        (0–30 pts)
    - Strategy-regime fit      (0–20 pts)
    - Confluence count         (0–20 pts)  how many strategies agree
    - LSTM alignment           (0–15 pts)
    - Risk/reward ratio        (0–15 pts)

    Grade: A (80–100), B (60–79), C (40–59), D (<40)
    """

    REGIME_FIT = {
        "Strong Trend":                 ["Trend_Following", "Momentum", "Breakout"],
        "Ranging / Choppy":             ["Mean_Reversion", "Scalping", "Arbitrage"],
        "High Volatility Breakout":     ["Breakout", "Momentum"],
        "Low Volatility Consolidation": ["Mean_Reversion", "Scalping"],
        "Unknown":                      [],
    }

    def score(
        self,
        signal:           dict,
        strategy_signals: dict,
        regime:           str,
        risk_plan:        dict,
        reasoning_engine = None,  # Added to access learner
    ) -> dict:
        pts = 0
        notes = []

        # 1. Raw confidence (max 30)
        conf  = signal.get("confidence", 0.0)
        source = signal.get("source", "")

        # Apply confidence multiplier from unsupervised learning
        multiplier = 1.0
        if reasoning_engine and reasoning_engine.learner:
            multiplier = reasoning_engine.learner.get_confidence_multiplier(source, regime)

        adjusted_conf = min(conf * multiplier, 1.0)  # Cap at 100%
        s_conf = round(adjusted_conf * 30)
        pts  += s_conf
        notes.append(f"Signal confidence {conf:.0%} × {multiplier:.2f} = {adjusted_conf:.0%} → {s_conf}/30 pts")

        # 2. Regime fit (max 20)
        source     = signal.get("source", "")
        fit_strats = self.REGIME_FIT.get(regime, [])
        if source in fit_strats:
            pts += 20
            notes.append(f"{source} fits '{regime}' regime → +20 pts")
        elif fit_strats:
            pts += 8
            notes.append(f"{source} is suboptimal for '{regime}' → +8 pts")
        else:
            pts += 12
            notes.append(f"Regime '{regime}' unknown → neutral +12 pts")

        # 3. Confluence — how many strategies agree with the direction (max 20)
        direction = signal.get("action", "WAIT")
        agreements = sum(
            1 for s in strategy_signals.values()
            if s.get("action") == direction and s.get("confidence", 0) > 0.3
        )
        total_strats = max(len(strategy_signals), 1)
        conf_score   = round((agreements / total_strats) * 20)
        pts         += conf_score
        notes.append(f"{agreements}/{total_strats} strategies agree → {conf_score}/20 pts")

        # 4. LSTM alignment (max 15)
        lstm = signal.get("lstm_prediction", {})
        lstm_dir = lstm.get("direction", "NEUTRAL")
        lstm_conf = lstm.get("confidence", 0.0)
        if (direction == "BUY" and lstm_dir == "UP") or \
           (direction == "SELL" and lstm_dir == "DOWN"):
            lstm_pts = round(lstm_conf * 15)
            pts += lstm_pts
            notes.append(f"LSTM confirms ({lstm_dir} @ {lstm_conf:.0%}) → +{lstm_pts} pts")
        elif lstm_dir == "NEUTRAL":
            pts += 7
            notes.append("LSTM neutral → +7 pts")
        else:
            notes.append(f"LSTM disagrees ({lstm_dir}) → +0 pts")

        # 5. Risk/reward quality (max 15)
        sl   = risk_plan.get("stop_loss_pips", 0)
        tp   = risk_plan.get("stop_loss_pips", 0) * 2  # assumed 1:2
        if sl > 0 and tp > 0:
            rr = tp / sl
            rr_pts = min(round(rr * 5), 15)
            pts   += rr_pts
            notes.append(f"R:R estimated 1:{rr:.1f} → {rr_pts}/15 pts")
        else:
            pts += 7
            notes.append("R:R not calculable → +7 pts")

        # Grade
        if pts >= 80:   grade = "A"
        elif pts >= 60: grade = "B"
        elif pts >= 40: grade = "C"
        else:           grade = "D"

        return {
            "score":   pts,
            "grade":   grade,
            "notes":   notes,
            "proceed": pts >= 40,
        }

    def verdict_sentence(self, scored: dict, symbol: str, direction: str) -> str:
        grade = scored["grade"]
        score = scored["score"]
        templates = {
            "A": [
                f"Quality looks solid on {symbol} — grade A ({score}/100). Proceeding.",
                f"{symbol} {direction} is a high-quality setup ({score} pts). Green light.",
                f"Strong confluence on {symbol}, grade A. Risk is well-defined.",
            ],
            "B": [
                f"{symbol} {direction} passes the bar — grade B ({score}/100). Reasonable setup.",
                f"Grade B on {symbol}. Not perfect, but the risk/reward justifies the entry.",
                f"{score}/100 on {symbol} — workable setup, manage size carefully.",
            ],
            "C": [
                f"Marginal setup on {symbol} — grade C ({score}/100). Proceed at reduced size.",
                f"{symbol} {direction} is a weak signal. Grade C — consider waiting.",
                f"Lukewarm confluence on {symbol} ({score}/100). Enter only if conviction is high.",
            ],
            "D": [
                f"Low-quality setup on {symbol} — grade D ({score}/100). Not recommended.",
                f"{score}/100 on {symbol}. I'd pass on this one.",
                f"Grade D on {symbol}. Too many factors working against this entry.",
            ],
        }
        return random.choice(templates.get(grade, templates["C"]))


# ── Risk commentary ───────────────────────────────────────────────────────────

class RiskCommentator:
    """Generates context-specific risk observations from live account data."""

    def comment(self, account: dict, config: dict) -> Optional[str]:
        """
        Returns a risk observation string if something noteworthy is detected.
        Returns None if everything looks healthy.
        """
        balance      = account.get("balance", 0)
        equity       = account.get("equity", 0)
        margin_level = account.get("margin_level", 999)
        floating_pnl = equity - balance
        max_loss     = config.get("max_daily_loss", 0)
        risk_pct     = config.get("risk_percentage", 1.0)

        # Drawdown check
        if balance > 0:
            drawdown_pct = (balance - equity) / balance * 100
            if drawdown_pct >= 5:
                return (
                    f"Floating drawdown is at {drawdown_pct:.1f}% (${abs(floating_pnl):,.2f}). "
                    f"You're using {drawdown_pct / (max_loss / balance * 100):.0%} of "
                    f"your daily loss budget."
                    if max_loss > 0 else
                    f"Floating drawdown at {drawdown_pct:.1f}% — watch the open positions."
                )

        # Margin level check
        if 0 < margin_level < 200:
            severity = "critical" if margin_level < 150 else "elevated"
            return (
                f"Margin level is {margin_level:.0f}% — {severity}. "
                f"New positions will increase liquidation risk."
            )

        # Floating profit — reinforce good behaviour
        if floating_pnl > 0 and balance > 0:
            gain_pct = floating_pnl / balance * 100
            if gain_pct >= 2:
                return (
                    f"Running ${floating_pnl:,.2f} in floating profit (+{gain_pct:.1f}%). "
                    f"Consider trailing stops to protect gains."
                )

        return None

    def overtrading_check(self, recent_trades: list) -> Optional[str]:
        """Detects concentrated or excessive trading patterns."""
        if len(recent_trades) < 5:
            return None

        symbols = [t.get("Symbol", t.get("symbol", "")) for t in recent_trades[-20:]]
        counts  = Counter(s for s in symbols if s)
        if not counts:
            return None

        top_sym, freq = counts.most_common(1)[0]
        if freq >= 8:
            msgs = [
                f"⚠️ {freq} trades on {top_sym} in recent history — that's heavy concentration. "
                f"Diversifying reduces correlated drawdown risk.",
                f"Noticing {freq} consecutive entries on {top_sym}. "
                f"Averaging into a losing direction? Check the open P&L.",
                f"Heavy {top_sym} exposure ({freq} trades). "
                f"If this is a recovery attempt, it rarely works — consider a hard stop.",
            ]
            return random.choice(msgs)

        # High overall trade frequency in last 10
        if len(recent_trades) >= 10:
            last_10 = recent_trades[-10:]
            buys  = sum(1 for t in last_10 if str(t.get("Action","")).upper() == "BUY")
            sells = sum(1 for t in last_10 if str(t.get("Action","")).upper() == "SELL")
            if buys >= 8 or sells >= 8:
                direction = "long" if buys >= 8 else "short"
                return (
                    f"Last 10 trades are heavily {direction}-biased. "
                    f"Bias can cloud risk management — make sure this is strategy-driven."
                )
        return None


# ── Regime advisor ────────────────────────────────────────────────────────────

class RegimeAdvisor:
    """Maps regime to actionable strategy guidance."""

    _ADVICE = {
        "Strong Trend": [
            "Trend is your friend here — ride momentum, keep stops loose enough to breathe.",
            "Strong trending regime. Mean reversion entries are a trap right now.",
            "Momentum and Trend Following are well-suited to current conditions.",
        ],
        "Ranging / Choppy": [
            "Market is chopping — favour mean reversion, tight targets, quick profits.",
            "Ranging conditions. Breakout trades will likely get faded.",
            "Low directional conviction in price action. Scalping and reversion outperform here.",
        ],
        "High Volatility Breakout": [
            "High-volatility environment — breakout trades are live, but expect whipsaws.",
            "Volatility is elevated. Size down and let the range resolve before trading.",
            "Breakout conditions detected. Watch for false breaks before committing.",
        ],
        "Low Volatility Consolidation": [
            "Market is coiling. Wait for the expansion — or scalp the tight range carefully.",
            "Low-vol squeeze building. Mean reversion has edge, but targets must be small.",
            "Consolidation regime. The bigger move is coming — patience pays here.",
        ],
        "Unknown": [
            "Regime is still being assessed — not enough data yet for a clear read.",
            "Not enough market history to classify regime. Use default risk settings.",
        ],
    }

    def advise(self, regime: str, reasoning_engine=None) -> str:
        options = self._ADVICE.get(regime, self._ADVICE["Unknown"])
        base_advice = random.choice(options)

        # Add insights from unsupervised learning if available
        if reasoning_engine and reasoning_engine.learner:
            insights = reasoning_engine.learner.generate_insights()
            if insights:
                # Add one relevant insight to the advice
                relevant_insights = [i for i in insights if regime.lower() in i.lower() or "regime" in i.lower()]
                if relevant_insights:
                    base_advice += f" {relevant_insights[0]}"

        return base_advice


# ── Response composer ─────────────────────────────────────────────────────────

class ResponseComposer:
    """
    Assembles multi-sentence responses from component parts.
    All responses are data-driven — no generic filler.
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

    def proactive(self, context: dict, regime: str = "Unknown") -> str:
        symbols   = context.get("tracked_symbols", [])
        n_symbols = len(symbols) if symbols else 0
        symbol    = context.get("last_symbol", "")
        pnl_str   = context.get("floating_pnl", "$0").replace("$", "").replace(",", "")
        try:
            pnl = float(pnl_str)
        except ValueError:
            pnl = 0.0

        from manager.reasoning_engine import RegimeAdvisor
        from manager.unsupervised_learner import STRATEGY_REGIME_FIT
        n_fit = len(STRATEGY_REGIME_FIT.get(regime, []))

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


# ── Main reasoning engine ─────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Orchestrates all reasoning components into coherent analysis.
    This is the main interface for the chatbot.
    """

    def __init__(self, strategy_manager, risk_manager, portfolio_manager):
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager

        # Initialize unsupervised learning engine (from strategy manager if available)
        self.learner = getattr(strategy_manager, 'learner', None) if strategy_manager else None

        # Initialize reasoning components
        self.narrator = SignalNarrator()
        self.reasoner = TradeReasoner()
        self.risk_commentator = RiskCommentator()
        self.regime_advisor = RegimeAdvisor()
        self.composer = ResponseComposer()

    def analyze_asset(self, symbol: str) -> dict:
        """Comprehensive asset analysis with scoring and narration."""
        # Get comprehensive signal analysis
        signal_data = self.strategy_manager.check_signals(symbol, use_ensemble=True)

        # Extract key components
        signal = {
            "action": signal_data.get("action", "WAIT"),
            "confidence": signal_data.get("confidence", 0.0),
            "reason": signal_data.get("reason", ""),
            "source": signal_data.get("source", "strategy engine"),
            "lstm_prediction": signal_data.get("lstm_prediction", {}),
        }
        strategy_signals = signal_data.get("strategy_signals", {})

        # Determine market regime
        regime = self._determine_market_regime(signal_data)

        # Get risk plan for scoring
        risk_plan = self._get_risk_plan(symbol)

        # Score the trade quality
        scored = self.reasoner.score(signal, strategy_signals, regime, risk_plan, self)

        # Generate narration
        narration = self.narrator.narrate(signal, symbol)

        # Get regime advice
        regime_advice = self.regime_advisor.advise(regime, self)

        return {
            "symbol": symbol.upper(),
            "signal": signal,
            "regime": regime,
            "scored": scored,
            "narration": narration,
            "regime_advice": regime_advice,
            "risk_plan": risk_plan,
        }

    def get_portfolio_context(self) -> dict:
        """Aggregates portfolio data for response injection."""
        try:
            account = self.portfolio_manager.broker.getAccountInfo()
            if account:
                total_value = account.equity
                daily_pnl = account.equity - account.balance
            else:
                total_value = 0.0
                daily_pnl = 0.0

            positions = self.portfolio_manager.broker.getPositions()
            active_positions = len(positions) if positions else 0

            # Calculate utilization (simplified - positions vs max allowed)
            max_trades = self.risk_manager.max_open_trades
            utilization = int((active_positions / max_trades) * 100) if max_trades > 0 else 0

            return {
                "total_value": f"{total_value:,.2f}",
                "active_positions": active_positions,
                "utilization": utilization,
                "daily_pnl": f"{daily_pnl:+,.2f}"
            }
        except Exception as e:
            # Fallback values
            return {
                "total_value": "0.00",
                "active_positions": 0,
                "utilization": 0,
                "daily_pnl": "+0.00"
            }

    def _determine_market_regime(self, signal_data: dict) -> str:
        """Determine the current market regime using unsupervised learning if available."""
        # Use unsupervised learner if available
        if self.learner:
            # Get current regime from the learner
            regime = self.learner.get_current_regime()
            if regime != "Unknown":
                return regime

        # Fallback to rule-based regime detection
        lstm_conf = signal_data.get('lstm_prediction', {}).get('confidence', 0.0)
        strategy_signals = signal_data.get('strategy_signals', {})

        # Check for consensus across strategies
        bullish_signals = sum(1 for sig in strategy_signals.values()
                            if sig.get('action') in ['BUY', 'BULLISH'])
        total_signals = len(strategy_signals)

        if lstm_conf > 0.7 and bullish_signals > total_signals * 0.6:
            return "Strong Trend"
        elif lstm_conf > 0.7 and bullish_signals < total_signals * 0.4:
            return "Ranging / Choppy"
        elif lstm_conf < 0.5:
            return "High Volatility Breakout"
        else:
            return "Low Volatility Consolidation"

    def _get_risk_plan(self, symbol: str) -> dict:
        """Get risk management plan for the symbol."""
        try:
            # This would normally call risk_manager.calculate_safe_trade
            # For now, return a mock plan
            return {
                "stop_loss_pips": 20,
                "take_profit_pips": 40,
                "lot_size": 0.01,
            }
        except:
            return {
                "stop_loss_pips": 20,
                "take_profit_pips": 40,
                "lot_size": 0.01,
            }