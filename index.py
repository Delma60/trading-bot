"""
bot_improvements.py — Five self-contained patches for ARIA.

Apply in order:
  1. OHLCVCache        → strategies/strategy_manager.py
  2. Bug fixes         → manager/agent_core.py
  3. TradeGatekeeper   → manager/risk_manager.py
  4. CorrelationGuard  → manager/risk_manager.py
  5. Auto-train hook   → strategies/strategy_manager.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — OHLCVCache
# File: strategies/strategy_manager.py
#
# Problem: Every strategy.analyze() call fetches fresh OHLCV data from MT5.
#   With 7 strategies and a 3-second scanner, that's ~140 broker API calls/min.
#   MT5 will silently start returning None or stale data under that load.
#
# Fix: Thread-safe LRU cache with a configurable TTL (default 60 s).
#   All strategies that run on the same bar share one fetch.
#
# Where: Add this class near the top of strategy_manager.py, then wire it
#   into check_signals() as shown below.
# ═══════════════════════════════════════════════════════════════════════════════

import threading
import time


class OHLCVCache:
    """
    Thread-safe time-to-live cache for OHLCV DataFrames.

    Prevents redundant MT5 calls when multiple strategies analyse the
    same symbol in the same scan cycle.

    Usage in StrategyManager.__init__:
        self._ohlcv_cache = OHLCVCache(ttl_seconds=60)

    Usage in check_signals() — replace the broker.ohclv_data() call:
        raw_df = self._ohlcv_cache.fetch(
            self.broker, symbol, timeframe, num_bars=1000
        )
    """

    def __init__(self, ttl_seconds: int = 60):
        self._store: dict[tuple, tuple[float, object]] = {}
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()

    def fetch(self, broker, symbol: str, timeframe: str, num_bars: int = 1000):
        key = (symbol.upper(), timeframe)
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.monotonic() - entry[0]) < self._ttl:
                return entry[1]   # cache hit

        # Cache miss — fetch from broker (outside the lock so we don't block)
        df = broker.ohclv_data(symbol, timeframe=timeframe, num_bars=num_bars)

        if df is not None and not df.empty:
            with self._lock:
                self._store[key] = (time.monotonic(), df)

        return df

    def invalidate(self, symbol: str = None, timeframe: str = None):
        """Force a refresh on next fetch. Call after known market events."""
        with self._lock:
            if symbol is None:
                self._store.clear()
            else:
                key = (symbol.upper(), timeframe or "H1")
                self._store.pop(key, None)


# ── Wire-up instructions for StrategyManager ──────────────────────────────────
#
# In StrategyManager.__init__, add:
#     self._ohlcv_cache = OHLCVCache(ttl_seconds=60)
#
# In StrategyManager.check_signals(), replace:
#     raw_df = self.broker.ohclv_data(symbol, timeframe=timeframe, num_bars=1000)
# with:
#     raw_df = self._ohlcv_cache.fetch(self.broker, symbol, timeframe, num_bars=1000)
#
# Also add to PortfolioManager.evaluate_portfolio_opportunities():
#   Pass the strategy_manager's cache instead of calling broker.ohclv_data directly
#   inside _get_current_market_state():
#     df = self.strategy_manager._ohlcv_cache.fetch(
#         self.broker, symbol, timeframe="H1", num_bars=50
#     )
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — Bug fixes in agent_core.py
#
# Bug A: htf_trend_check step is referenced inside _quality_score() but is
#   never added to any AgentPlan.steps list.  Result: it always returns
#   {"trend": "NEUTRAL"}, silently zeroing out the HTF alignment score.
#
# Bug B: AgentSynthesizer.synthesize() only handles "analyze_symbol" and
#   "greeting" — every other intent (account_summary, active_positions,
#   bulk_scan, trade_history, risk_management, etc.) returns the placeholder
#   string "Action '{intent}' processed successfully."
#
# Bug C: plan.suggested_action is never written by the synthesizer, so
#   the _maybe_execute() confirm-trade flow in chat.py never triggers.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Bug A fix: add htf_trend_check to trade and analysis plans ────────────────
#
# In AgentPlanner._plan_analysis(), change steps list to:
#   steps=[
#       AgentStep("market_regime",    f"Detecting market regime for {symbol}"),
#       AgentStep("htf_trend_check",  f"Higher-timeframe trend — {symbol}"),   # ADD
#       AgentStep("signal_ensemble",  f"Running strategy ensemble on {symbol}"),
#       AgentStep("quality_score",    "Scoring signal quality"),
#       AgentStep("risk_check",       "Risk gate validation"),
#       AgentStep("position_sizing",  "Calculating optimal lot size"),
#       AgentStep("anomaly_check",    "Checking for anomalous conditions"),
#       AgentStep("correlation_hint", "Scanning correlated pairs"),
#   ]
#
# Apply the same addition to _plan_trade():
#   steps=[
#       AgentStep("market_regime",   f"Market regime check — {symbol}"),
#       AgentStep("htf_trend_check", f"Higher-timeframe trend — {symbol}"),    # ADD
#       AgentStep("signal_ensemble", f"Signal validation — {symbol}"),
#       AgentStep("quality_score",   "Signal quality scoring"),
#       AgentStep("risk_check",      "Risk gate validation"),
#       AgentStep("position_sizing", "Position sizing calculation"),
#       AgentStep("anomaly_check",   "Anomaly detection"),
#   ]
#
# Also add "htf_trend_check" to AgentExecutor._dispatch() handlers dict:
#   "htf_trend_check": lambda: self._htf_trend_check(sym),
# ─────────────────────────────────────────────────────────────────────────────


# ── Bug B fix: rewrite AgentSynthesizer.synthesize() ─────────────────────────
#
# Replace the entire AgentSynthesizer class with the one below.
# It reads actual step results from plan.context and builds real responses
# for every intent the agent handles.
# ─────────────────────────────────────────────────────────────────────────────

import random


class AgentSynthesizerFixed:
    """
    Drop-in replacement for AgentSynthesizer in manager/agent_core.py.

    Reads actual step results from plan.steps (via _result()) and builds
    data-grounded responses for every intent.  No more generic placeholders.
    """

    def __init__(self, reasoning):
        self.reasoning = reasoning

    def synthesize(self, plan) -> str:
        intent = plan.intent
        r = {step.name: (step.result or {}) for step in plan.steps}

        dispatch = {
            "analyze_symbol":    self._analysis,
            "ai_analysis":       self._analysis,
            "execute_trade":     self._analysis,    # pre-execution analysis view
            "open_buy":          self._analysis,
            "open_sell":         self._analysis,
            "greeting":          self._greeting,
            "account_summary":   self._account,
            "active_positions":  self._positions,
            "profitable_positions": self._positions,
            "bulk_scan":         self._scan,
            "risk_management":   self._risk,
            "trade_history":     self._history,
            "get_price":         self._price,
            "close_position":    self._close_confirm,
            "close_all":         self._close_all_confirm,
            "news_update":       self._news,
            "strategy_info":     self._strategy,
            "portfolio_status":  self._portfolio,
        }

        fn = dispatch.get(intent, self._general)
        try:
            return fn(plan, r)
        except Exception:
            return self._general(plan, r)

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _analysis(self, plan, r) -> str:
        symbol  = plan.symbol or "the asset"
        sig     = r.get("signal_ensemble", {})
        scored  = r.get("quality_score", {})
        regime  = r.get("market_regime", {}).get("regime", "Unknown")
        sizing  = r.get("position_sizing", {})
        htf     = r.get("htf_trend_check", {}).get("trend", "NEUTRAL")
        anomaly = r.get("anomaly_check", {})

        action  = sig.get("action", "WAIT")
        conf    = sig.get("confidence", 0.0)
        reason  = sig.get("reason", "")
        grade   = scored.get("grade", "?")
        notes   = scored.get("notes", [])
        lots    = sizing.get("lots")
        approved= sizing.get("approved", False)

        # Direction opener
        openers = {
            "BUY":  [f"Bullish confluence on {symbol}.", f"{symbol} is flagging a long entry."],
            "SELL": [f"Bearish distribution on {symbol}.", f"{symbol} is rolling over short."],
            "WAIT": [f"No clear edge on {symbol} right now.", f"Staying flat on {symbol}."],
        }
        opener = random.choice(openers.get(action, openers["WAIT"]))

        # Confidence qualifier
        conf_word = ("high-conviction" if conf > 0.75
                     else "moderate" if conf > 0.55
                     else "tentative" if conf > 0.40
                     else "weak")

        parts = [f"{opener} {conf_word} signal at {conf:.0%}."]

        # Regime
        parts.append(f"Regime: {regime}.")

        # HTF alignment
        if action != "WAIT":
            if htf == action:
                parts.append(f"HTF trend aligned ({htf}) — good.")
            elif htf not in ("NEUTRAL", "WAIT"):
                parts.append(f"HTF trend is {htf} — counter-trend entry, be cautious.")

        # Top scoring note
        if notes:
            clean = notes[0].split(" →")[0].strip().rstrip(".")
            parts.append(f"{clean}.")

        # Grade verdict
        grade_lines = {
            "A": "Grade A — premium setup.",
            "B": "Grade B — solid, proceed with standard size.",
            "C": "Grade C — marginal. Consider waiting.",
            "D": "Grade D — skip this one.",
        }
        parts.append(grade_lines.get(grade, f"Grade {grade}."))

        # Anomaly warning
        if anomaly.get("is_anomaly"):
            parts.append(f"Anomaly score {anomaly['anomaly_score']:.2f} — unusual conditions, size down.")

        # Suggested action for confirm flow (Bug C fix)
        if action in ("BUY", "SELL") and grade in ("A", "B") and approved and lots:
            plan.suggested_action = f"{action} {lots} {symbol}"

        return " ".join(parts)

    def _greeting(self, plan, r) -> str:
        acct = r.get("account_data", {})
        pos  = r.get("open_positions", {})
        reg  = r.get("regime_snapshot", {}).get("regime", "Unknown")

        if acct:
            bal  = acct.get("balance", 0)
            eq   = acct.get("equity", 0)
            pnl  = acct.get("profit", 0)
            n    = pos.get("count", 0)
            sign = "+" if pnl >= 0 else ""
            base = (f"Back online. Balance ${bal:,.2f}, equity ${eq:,.2f} "
                    f"({sign}${pnl:,.2f} floating). {n} open position(s).")
        else:
            base = "Back online. Broker connected."

        suggestions = [
            "Run a portfolio scan?",
            "Check positions, or scan for fresh setups?",
            "Ready when you are — scan, analyse, or check risk?",
        ]
        return f"{base} Regime: {reg}. {random.choice(suggestions)}"

    def _account(self, plan, r) -> str:
        acct = r.get("account_data", {})
        dd   = r.get("drawdown_analysis", {})
        fl   = r.get("floating_pnl", {})

        if not acct:
            return "Could not fetch account data from MT5."

        bal  = acct.get("balance", 0)
        eq   = acct.get("equity", 0)
        ml   = acct.get("margin_level")
        fm   = acct.get("free_margin", 0)
        dd_v = dd.get("trailing_drawdown", 0)
        dd_p = dd.get("drawdown_pct", 0)
        pnl  = fl.get("total", acct.get("profit", 0))

        ml_str = f"{ml:.1f}%" if ml else "N/A"
        lines = [
            f"Balance ${bal:,.2f} | Equity ${eq:,.2f} | Free margin ${fm:,.2f}.",
            f"Margin level {ml_str}.",
            f"Floating P&L ${pnl:+,.2f}.",
        ]
        if dd_v > 0:
            lines.append(f"Trailing drawdown from peak: ${dd_v:,.2f} ({dd_p:.1f}%).")
        return " ".join(lines)

    def _positions(self, plan, r) -> str:
        pos_data = r.get("open_positions", {})
        positions = pos_data.get("positions", [])

        if not positions:
            return "No open positions."

        total = pos_data.get("total_pnl", 0)
        lines = [f"{len(positions)} open position(s). Total floating: ${total:+,.2f}."]
        for p in positions:
            pnl_sign = "+" if p["profit"] >= 0 else ""
            lines.append(
                f"  {p['symbol']} {p['action']} {p['volume']}L "
                f"@ {p['price_open']} → {pnl_sign}${p['profit']:.2f}"
            )
        return "\n".join(lines)

    def _scan(self, plan, r) -> str:
        best  = r.get("best_opportunities", {})
        gate  = r.get("risk_gate", {})
        regime= r.get("regime_snapshot", {}).get("regime", "Unknown")

        if not gate.get("gate_open", True):
            return f"Risk gate closed — {gate.get('reason', 'drawdown limit reached')}. No scan executed."

        executed = best.get("executed", [])
        waiting  = best.get("waiting", [])
        blocked  = best.get("blocked", [])
        total    = best.get("total", 0)

        parts = [f"Portfolio scan complete. {total} symbol(s) checked. Regime: {regime}."]

        if executed:
            parts.append(f"{len(executed)} trade(s) executed:")
            parts.extend(f"  {e}" for e in executed)
        if waiting:
            parts.append(f"{len(waiting)} symbol(s) waiting for a signal.")
        if blocked:
            parts.append(f"{len(blocked)} blocked by risk rules.")
        if not executed and not waiting:
            parts.append("No actionable setups found this cycle.")

        return "\n".join(parts)

    def _risk(self, plan, r) -> str:
        acct = r.get("account_data", {})
        dd   = r.get("drawdown_analysis", {})
        mg   = r.get("margin_check", {})
        ot   = r.get("overtrading_check", {})
        pos  = r.get("open_positions", {})

        parts = []
        if acct:
            bal = acct.get("balance", 0)
            eq  = acct.get("equity", 0)
            parts.append(f"Balance ${bal:,.2f} | Equity ${eq:,.2f}.")

        if dd:
            dd_v = dd.get("trailing_drawdown", 0)
            dd_p = dd.get("drawdown_pct", 0)
            hi   = dd.get("daily_high", 0)
            parts.append(
                f"Peak equity today ${hi:,.2f}. "
                f"Trailing drawdown ${dd_v:,.2f} ({dd_p:.1f}%)."
            )

        ml = mg.get("margin_level")
        if ml:
            severity = "critical" if ml < 150 else "healthy" if ml > 300 else "moderate"
            parts.append(f"Margin level {ml:.0f}% — {severity}.")

        n_pos = pos.get("count", 0)
        parts.append(f"{n_pos} open position(s).")

        ot_warn = ot.get("warning")
        if ot_warn:
            parts.append(f"Concentration warning: {ot_warn}")

        return " ".join(parts) if parts else "Unable to retrieve risk metrics."

    def _history(self, plan, r) -> str:
        hist = r.get("closed_trades", {})
        perf = r.get("performance_analysis", {})

        if not hist:
            return "No trade history data available."

        count  = hist.get("count", 0)
        pnl    = hist.get("pnl", 0)
        buys   = hist.get("buys", 0)
        sells  = hist.get("sells", 0)
        closes = hist.get("closes", 0)

        if count == 0:
            return "No closed trades recorded today."

        lines = [
            f"{count} trade action(s) today: {buys} buy, {sells} sell, {closes} close.",
            f"Realised P&L: ${pnl:+,.2f}.",
        ]

        best  = perf.get("best_performer")
        worst = perf.get("worst_performer")
        by_sym= perf.get("by_symbol", {})

        if best and by_sym.get(best, 0) > 0:
            lines.append(f"Best: {best} (+${by_sym[best]:.2f}).")
        if worst and by_sym.get(worst, 0) < 0:
            lines.append(f"Worst: {worst} (${by_sym[worst]:.2f}).")

        return " ".join(lines)

    def _price(self, plan, r) -> str:
        tick   = r.get("live_price", {})
        regime = r.get("market_regime", {}).get("regime", "Unknown")
        symbol = plan.symbol or tick.get("symbol", "?")

        bid    = tick.get("bid", 0)
        ask    = tick.get("ask", 0)
        spread = tick.get("spread", 0)

        if not bid:
            return f"Could not fetch live price for {symbol}."

        return (
            f"{symbol}: Bid {bid} / Ask {ask}. "
            f"Spread {spread:.1f} pips. Regime: {regime}."
        )

    def _close_confirm(self, plan, r) -> str:
        pos  = r.get("open_positions", {})
        sym  = plan.symbol or "unknown"
        poss = [p for p in pos.get("positions", []) if p["symbol"] == sym]
        if not poss:
            return f"No open position found for {sym}."
        p = poss[0]
        return (
            f"Ready to close {sym} ({p['action']} {p['volume']}L, "
            f"floating ${p['profit']:+.2f}). Say 'yes' to confirm."
        )

    def _close_all_confirm(self, plan, r) -> str:
        pos   = r.get("open_positions", {})
        fl    = r.get("floating_pnl", {})
        n     = pos.get("count", 0)
        total = fl.get("total", 0)
        if n == 0:
            return "No open positions to close."
        return (
            f"Close all {n} position(s)? Total floating: ${total:+,.2f}. "
            f"Say 'yes' to confirm."
        )

    def _news(self, plan, r) -> str:
        regime = r.get("regime_snapshot", {}).get("regime", "Unknown")
        acct   = r.get("account_data", {})
        eq     = acct.get("equity", 0) if acct else 0
        return (
            f"Current regime: {regime}. "
            f"No live news feed connected — consider checking ForexFactory or Reuters "
            f"for upcoming events before trading. "
            f"Equity: ${eq:,.2f}."
        )

    def _strategy(self, plan, r) -> str:
        snap  = r.get("regime_snapshot", {})
        regime= snap.get("regime", "Unknown")
        best  = snap.get("best_strategies", [])
        return (
            f"Current regime: {regime}. "
            f"Best-fit strategies: {', '.join(best) if best else 'undetermined'}. "
            f"Ensemble runs Mean Reversion, Momentum, Breakout, Scalping, Arbitrage, "
            f"and Trend Following simultaneously. MetaScorer weights their votes "
            f"using a trained XGBoost model (falls back to weighted voting until trained)."
        )

    def _portfolio(self, plan, r) -> str:
        acct = r.get("account_data", {})
        pos  = r.get("open_positions", {})
        health = r.get("portfolio_health", {}).get("health_summary", "")
        eq   = acct.get("equity", 0) if acct else 0
        bal  = acct.get("balance", 0) if acct else 0
        n    = pos.get("count", 0)
        pnl  = pos.get("total_pnl", 0)
        daily= eq - bal
        return (
            f"Portfolio equity ${eq:,.2f}. "
            f"{n} open position(s), floating ${pnl:+,.2f}. "
            f"Daily P&L ${daily:+,.2f}. "
            + (f"{health}" if health else "")
        )

    def _general(self, plan, r) -> str:
        snap = r.get("context_snapshot", {})
        bal  = snap.get("balance", 0)
        n    = snap.get("open_positions", 0)
        reg  = snap.get("regime", "Unknown")
        return (
            f"Balance ${bal:,.2f} | {n} open position(s) | Regime: {reg}. "
            f"Try: scan, positions, account, or name a symbol."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — TradeGatekeeper
# File: manager/risk_manager.py  (add class, then call from
#        PortfolioManager.evaluate_portfolio_opportunities and
#        ActionExecutor.execute_trade)
#
# Problem: The bot will trade at 2 AM with a 4-pip spread on a dead market.
#   No check is made on spread width or trading session quality before
#   submitting an order.  Both silently destroy edge.
#
# Fix: A single gate() call returns (allowed, reason) before execution.
# ═══════════════════════════════════════════════════════════════════════════════

import MetaTrader5 as mt5
from datetime import datetime, timezone


class TradeGatekeeper:
    """
    Pre-trade gate that checks spread and session quality.

    Call gate() before any order submission.  Returns (True, "OK") when
    conditions are acceptable, or (False, reason) to block the trade.

    Parameters
    ----------
    max_spread_pips   : Maximum allowed spread in pips.  Signals above this
                        are likely to be unprofitable after spread cost.
                        Default 3.0 for major forex pairs.
    avoid_asian_session: Block trades between 00:00–07:00 UTC, when
                        majors have wide spreads and thin liquidity.
    avoid_friday_close: Block new trades after 20:00 UTC on Fridays
                        (weekend gap risk).
    """

    def __init__(
        self,
        max_spread_pips:     float = 3.0,
        avoid_asian_session: bool  = True,
        avoid_friday_close:  bool  = True,
    ):
        self.max_spread_pips     = max_spread_pips
        self.avoid_asian_session = avoid_asian_session
        self.avoid_friday_close  = avoid_friday_close

    def gate(self, symbol: str, broker) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call this before submitting any order.
        """
        # ── Session filter ───────────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        hour    = now_utc.hour
        weekday = now_utc.weekday()   # 0=Mon … 4=Fri … 6=Sun

        if self.avoid_asian_session and 0 <= hour < 7:
            return False, (
                f"Asian session ({hour:02d}:00 UTC) — low liquidity, wide spreads. "
                f"Waiting for London open."
            )

        if self.avoid_friday_close and weekday == 4 and hour >= 20:
            return False, (
                "Friday after 20:00 UTC — weekend gap risk. "
                "No new positions before market close."
            )

        if weekday >= 5:   # Saturday or Sunday
            return False, "Market closed (weekend)."

        # ── Spread filter ────────────────────────────────────────────────────
        spread_pips = self._get_spread_pips(symbol, broker)
        if spread_pips is None:
            return False, f"Could not fetch tick data for {symbol}."

        if spread_pips > self.max_spread_pips:
            return False, (
                f"Spread too wide: {spread_pips:.1f} pips "
                f"(limit {self.max_spread_pips} pips). "
                f"Waiting for tighter conditions."
            )

        return True, "OK"

    def _get_spread_pips(self, symbol: str, broker) -> float | None:
        try:
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if not tick or not info:
                return None
            spread_points = (tick.ask - tick.bid) / info.point
            # Convert points → pips (10 points per pip for most forex)
            pip_multiplier = 1.0 if any(
                t in symbol.upper()
                for t in ["BTC", "ETH", "XAU", "XAG", "US30", "NAS"]
            ) else 10.0
            return spread_points / pip_multiplier
        except Exception:
            return None


# ── Wire-up instructions ───────────────────────────────────────────────────────
#
# In PortfolioManager.__init__, add:
#     from manager.risk_manager import TradeGatekeeper
#     self.gate = TradeGatekeeper(max_spread_pips=3.0)
#
# In PortfolioManager.evaluate_portfolio_opportunities(), before signal check:
#     ok, gate_reason = self.gate.gate(symbol, self.broker)
#     if not ok:
#         results.append(f"⏳ {symbol}: {gate_reason}")
#         continue
#
# In ActionExecutor.execute_trade(), before self.broker.execute_trade():
#     from manager.risk_manager import TradeGatekeeper
#     gk = TradeGatekeeper()
#     ok, reason = gk.gate(symbol, None)   # broker not needed; uses mt5 directly
#     if not ok:
#         return f"Trade blocked by gate: {reason}"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 4 — CorrelationGuard
# File: manager/risk_manager.py
#
# Problem: The bot can simultaneously hold EURUSD, GBPUSD, and EURGBP.
#   These share USD and EUR legs — it's not three positions, it's three
#   bets on the same underlying.  A USD news spike will hit all three at once.
#
# Fix: Track the currency legs of every open position and refuse a new trade
#   if adding it would put more than max_shared_legs legs on the same currency.
# ═══════════════════════════════════════════════════════════════════════════════

from collections import Counter


class CorrelationGuard:
    """
    Prevents over-concentration in a single underlying currency.

    Works by decomposing each 6-character forex symbol into its two legs
    (e.g. EURUSD → EUR, USD) and counting how many open positions share a
    leg with the proposed new trade.

    Parameters
    ----------
    max_shared_legs : int
        Maximum number of times one currency may appear across open positions
        (including the proposed new trade).  Default 2 — allows one existing
        position on the same currency before blocking a new entry.
    """

    def __init__(self, max_shared_legs: int = 2):
        self.max_shared_legs = max_shared_legs

    def check(self, symbol: str, broker) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        Call before opening a new position.
        """
        positions = broker.getPositions() or []
        if not positions:
            return True, "No existing positions."

        # Extract legs from all open positions
        open_legs: list[str] = []
        for pos in positions:
            open_legs.extend(self._legs(pos.symbol))

        leg_counts = Counter(open_legs)

        # Check proposed symbol's legs against existing counts
        proposed_legs = self._legs(symbol)
        for leg in proposed_legs:
            # Count would become leg_counts[leg] + 1 after this trade
            if leg_counts[leg] + 1 > self.max_shared_legs:
                concentrated = [
                    pos.symbol for pos in positions
                    if leg in self._legs(pos.symbol)
                ]
                return False, (
                    f"Correlation limit: {leg} already appears in "
                    f"{leg_counts[leg]} open position(s) "
                    f"({', '.join(concentrated)}). "
                    f"Max shared legs = {self.max_shared_legs}."
                )

        return True, "OK"

    @staticmethod
    def _legs(symbol: str) -> list[str]:
        """Split a 6-char forex symbol into its two 3-char currency legs."""
        sym = symbol.upper()
        # Skip non-forex instruments (metals, crypto, indices)
        if any(sym.startswith(p) for p in ["XAU", "XAG", "BTC", "ETH", "US", "GER"]):
            return []
        if len(sym) >= 6:
            return [sym[:3], sym[3:6]]
        return []


# ── Wire-up instructions ───────────────────────────────────────────────────────
#
# In PortfolioManager.__init__, add:
#     from manager.risk_manager import CorrelationGuard
#     self.corr_guard = CorrelationGuard(max_shared_legs=2)
#
# In PortfolioManager.evaluate_portfolio_opportunities(), after the risk check:
#     ok, corr_reason = self.corr_guard.check(symbol, self.broker)
#     if not ok:
#         results.append(f"⚠️ {symbol}: {corr_reason}")
#         continue
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 5 — MetaScorer auto-training hook
# File: strategies/strategy_manager.py
#
# Problem: StrategyManager.check_signals() builds a feature vector and stores
#   it in the return dict as "feature_vector", but nothing ever calls
#   meta.collect_sample() or meta.train().  The XGBoost meta-model never
#   escapes weighted-voting mode, regardless of how many trades are taken.
#
# Fix: Wire the outcome of each closed position back into the MetaScorer,
#   and trigger training automatically once the sample threshold is reached.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Add this method to StrategyManager ────────────────────────────────────────

def record_trade_outcome(self, feature_vector, action: str, profit: float):
    """
    Call this when a position CLOSES.

    Parameters
    ----------
    feature_vector : np.ndarray
        The vector returned in check_signals()["feature_vector"] at the
        time the trade was opened.  Store it in a dict keyed by ticket
        when the trade opens; pass it here when it closes.
    action : str
        "BUY" or "SELL" — the direction that was taken.
    profit : float
        Final realised P&L of the closed position.

    The label passed to MetaScorer is:
        - the actual direction ("BUY"/"SELL") if profitable
        - "WAIT" if the trade was a loss (meaning the model should have
          waited, regardless of direction)
    """
    label = action if profit > 0 else "WAIT"
    self.meta.collect_sample(feature_vector, label)

    # Auto-train once we cross the minimum sample threshold
    sample_count = len(self.meta._samples)
    if sample_count >= 200 and sample_count % 50 == 0:
        # Every 50 new samples after the 200-sample threshold, retrain
        acc = self.meta.train(force=True)
        if acc > 0:
            self.notify(
                f"[MetaScorer] Retrained on {sample_count} samples. "
                f"Validation accuracy: {acc:.1%}",
                priority="normal",
            )


# ── Wire-up in PortfolioManager ───────────────────────────────────────────────
#
# In PortfolioManager.log_trade_for_learning(), replace the method body with:
#
#     trade_data = getattr(self, '_temporary_trade_states', {}).pop(symbol, None)
#     if not trade_data:
#         return
#
#     fv = np.array(trade_data["state"])   # feature vector saved at open
#     action = trade_data.get("action", "BUY")
#     self.strategy_manager.record_trade_outcome(fv, action, profit)
#
# And when executing a trade in evaluate_portfolio_opportunities(), save the
# action alongside the state:
#     self._temporary_trade_states[symbol] = {
#         "state":    current_state.tolist(),
#         "strategy": strategy_name,
#         "action":   signal["action"],        # ADD THIS LINE
#     }
# ─────────────────────────────────────────────────────────────────────────────