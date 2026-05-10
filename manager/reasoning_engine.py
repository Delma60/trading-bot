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
5. ResponseEngine   — assembles scored, varied, context-filled responses

Design principle: every output is *earned* — it references real numbers
from live context, not generic filler.
"""

import random
import math
from manager.response_engine import ResponseEngine
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
        reasoning_engine = None,
        htf_trend = "NEUTRAL"
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
            pts += 0
            notes.append("LSTM neutral → +0 pts")
        else:
            notes.append(f"LSTM disagrees ({lstm_dir}) → +0 pts")

        # 5. Risk/reward quality (max 15)
       # Improved R:R scoring logic in reasoning_engine.py
        sl_pips = risk_plan.get("stop_loss_pips", 0)
        tp_pips = risk_plan.get("take_profit_pips", sl_pips * 1.5) # Fallback to 1.5 if missing

        if sl_pips > 0 and tp_pips > 0:
            rr = tp_pips / sl_pips
            # Reward setups that have at least 1:1.5, penalize anything under 1:1
            if rr < 1.0:
                rr_pts = 0
                notes.append(f"Poor R:R (1:{rr:.1f}). SL is wider than TP target → 0/15 pts")
            else:
                rr_pts = min(round((rr - 1.0) * 10), 15)  # 1:2.5 gives max 15 points
                pts += rr_pts
                notes.append(f"Strong R:R profiling 1:{rr:.1f} → {rr_pts}/15 pts")

        # 6. Higher Timeframe (HTF) Alignment
        direction = signal.get("action", "WAIT")
        if direction in ["BUY", "SELL"]:
            if htf_trend == direction:
                pts += 15  # Bonus for trading with the macro trend
                notes.append(f"HTF Trend Aligned ({htf_trend}) → +15 pts")
            elif htf_trend == "NEUTRAL":
                pts += 5
                notes.append("HTF Trend is neutral → +5 pts")
            else:
                pts -= 15  # Severe penalty for fighting the macro trend
                notes.append(f"Counter HTF Trend (HTF is {htf_trend}) → -15 pts")

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
        self.composer = ResponseEngine()

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
        if not symbol or not self.risk_manager:
            return {
                "stop_loss_pips": 20,
                "take_profit_pips": 40,
                "lot_size": 0.01,
            }

        try:
            safe_trade = self.risk_manager.calculate_safe_trade(
                symbol=symbol,
                base_risk_pct=1.0,
                stop_loss_pips=20.0,
                max_daily_loss=500.0,
                portfolio_size=max(1, getattr(self.risk_manager, 'max_open_trades', 1)),
            )

            stop_loss = float(safe_trade.get("stop_loss_pips", 20.0))
            take_profit = float(safe_trade.get("take_profit_pips", stop_loss * 2))
            if take_profit <= 0:
                take_profit = stop_loss * 2

            return {
                "stop_loss_pips": stop_loss,
                "take_profit_pips": take_profit,
                "lot_size": float(safe_trade.get("lots", 0.01)),
            }
        except Exception:
            return {
                "stop_loss_pips": 20,
                "take_profit_pips": 40,
                "lot_size": 0.01,
            }