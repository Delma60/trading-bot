"""
manager/agent_core.py — ARIA's Agentic Reasoning Core

Multi-step planning → execution → synthesis pipeline that converts user
intent into contextualized, action-oriented trading decisions.

Architecture
-----------
AgentPlanner   : converts (intent, entities, memory) → ordered AgentPlan
AgentExecutor  : runs each step against live broker / ML systems
AgentSynthesizer: merges step results into a structured, data-rich response
AgentCore      : façade that wires the three components together

Design principles
-----------------
- Every output is *earned*: it references real numbers from live context.
- Each step is independent and logged; failures are non-fatal.
- Synthesis is deterministic: given the same step results it produces the
  same class of response, but varied phrasing.
- The planner determines WHAT to do; the synthesizer decides HOW to say it.
"""

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from manager.risk_manager import DynamicRiskTargeter


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    name: str
    description: str
    result: Optional[dict] = None
    status: str = "pending"   # pending | running | done | skipped | failed
    duration_ms: float = 0.0


@dataclass
class AgentPlan:
    intent: str
    symbol: Optional[str]
    steps: list
    context: dict = field(default_factory=dict)
    final_response: str = ""
    suggested_action: Optional[str] = None    # e.g. "execute BUY 0.01 EURUSD"
    action_params: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# AgentPlanner
# ─────────────────────────────────────────────────────────────────────────────

class AgentPlanner:
    """
    Maps an intent + entities to a deterministic sequence of AgentSteps.
    Steps are cheap to skip — the executor gracefully handles failures.
    """

    def plan(self, intent: str, entities: dict, memory: dict) -> AgentPlan:
        symbols = entities.get("symbols") or []
        symbol = symbols[0] if symbols else memory.get("last_symbol", "EURUSD")
        direction = entities.get("direction")

        plan_builders = {
            "analyze_symbol":   self._plan_analysis,
            "ai_analysis":      self._plan_analysis,
            "execute_trade":    self._plan_trade,
            "open_buy":         self._plan_trade,
            "open_sell":        self._plan_trade,
            "trade_execution":  self._plan_trade,
            "portfolio_status": self._plan_portfolio,
            "bulk_scan":        self._plan_scan,
            "risk_management":  self._plan_risk,
            "account_summary":  self._plan_account,
            "trade_history":    self._plan_history,
            "active_positions": self._plan_positions,
            "profitable_positions": self._plan_positions,
            "get_price":        self._plan_price_check,
            "close_position":   self._plan_close,
            "close_all":        self._plan_close_all,
            "news_update":      self._plan_news,
            "greeting":         self._plan_greeting,
            "strategy_info":    self._plan_strategy_info,
        }

        builder = plan_builders.get(intent, self._plan_general)
        plan = builder(symbol, entities, memory)
        plan.context["requested_direction"] = direction
        plan.context["entities"] = entities
        return plan

    # ── Plan templates ────────────────────────────────────────────────────────

    def _plan_analysis(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="analyze_symbol", symbol=symbol, steps=[
            AgentStep("market_regime",    f"Detecting market regime for {symbol}"),
            AgentStep("signal_ensemble",  f"Running strategy ensemble on {symbol}"),
            AgentStep("quality_score",    "Scoring signal quality"),
            AgentStep("risk_check",       "Risk gate validation"),
            AgentStep("position_sizing",  "Calculating optimal lot size"),
            AgentStep("anomaly_check",    "Checking for anomalous conditions"),
            AgentStep("correlation_hint", "Scanning correlated pairs"),
        ])

    def _plan_trade(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="execute_trade", symbol=symbol, steps=[
            AgentStep("market_regime",   f"Market regime check — {symbol}"),
            AgentStep("signal_ensemble", f"Signal validation — {symbol}"),
            AgentStep("quality_score",   "Signal quality scoring"),
            AgentStep("risk_check",      "Risk gate validation"),
            AgentStep("position_sizing", "Position sizing calculation"),
            AgentStep("anomaly_check",   "Anomaly detection"),
        ])

    def _plan_portfolio(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="portfolio_status", symbol=None, steps=[
            AgentStep("account_data",     "Fetching live account data"),
            AgentStep("open_positions",   "Reviewing open positions"),
            AgentStep("portfolio_health", "Portfolio health assessment"),
            AgentStep("regime_snapshot",  "Market regime snapshot"),
            AgentStep("drawdown_analysis","Drawdown analysis"),
        ])

    def _plan_scan(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="bulk_scan", symbol=None, steps=[
            AgentStep("risk_gate",          "Global risk gate check"),
            AgentStep("portfolio_scan",     "Scanning portfolio for opportunities"),
            AgentStep("best_opportunities", "Ranking top opportunities"),
            AgentStep("regime_snapshot",    "Regime context"),
        ])

    def _plan_risk(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="risk_management", symbol=None, steps=[
            AgentStep("account_data",      "Fetching account metrics"),
            AgentStep("drawdown_analysis", "Drawdown and watermark analysis"),
            AgentStep("overtrading_check", "Overtrading pattern detection"),
            AgentStep("margin_check",      "Margin level assessment"),
            AgentStep("open_positions",    "Current exposure review"),
        ])

    def _plan_account(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="account_summary", symbol=None, steps=[
            AgentStep("account_data",  "Fetching live account data"),
            AgentStep("floating_pnl",  "Floating P&L breakdown"),
            AgentStep("regime_snapshot","Market context"),
        ])

    def _plan_history(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="trade_history", symbol=None, steps=[
            AgentStep("closed_trades",       "Fetching today's trade log"),
            AgentStep("performance_analysis","Performance pattern analysis"),
            AgentStep("overtrading_check",   "Trade concentration check"),
        ])

    def _plan_positions(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="active_positions", symbol=None, steps=[
            AgentStep("open_positions",  "Fetching open positions"),
            AgentStep("risk_per_pos",    "Risk-per-position assessment"),
            AgentStep("portfolio_health","Portfolio health"),
        ])

    def _plan_price_check(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="get_price", symbol=symbol, steps=[
            AgentStep("live_price",     f"Fetching live quote — {symbol}"),
            AgentStep("market_regime",  f"Quick regime read — {symbol}"),
        ])

    def _plan_close(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="close_position", symbol=symbol, steps=[
            AgentStep("open_positions", f"Verifying open positions for {symbol}"),
        ])

    def _plan_close_all(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="close_all", symbol=None, steps=[
            AgentStep("open_positions", "Verifying all open positions"),
            AgentStep("floating_pnl",   "Final floating P&L snapshot"),
        ])

    def _plan_news(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="news_update", symbol=None, steps=[
            AgentStep("regime_snapshot", "Reading current market regime"),
            AgentStep("account_data",    "Account risk context"),
        ])

    def _plan_greeting(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="greeting", symbol=None, steps=[
            AgentStep("account_data",    "Account summary"),
            AgentStep("open_positions",  "Open positions check"),
            AgentStep("regime_snapshot", "Market regime"),
        ])

    def _plan_strategy_info(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="strategy_info", symbol=None, steps=[
            AgentStep("regime_snapshot", "Current regime and strategy fit"),
            AgentStep("account_data",    "Account context"),
        ])

    def _plan_general(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="general", symbol=None, steps=[
            AgentStep("context_snapshot", "Building context snapshot"),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# AgentExecutor
# ─────────────────────────────────────────────────────────────────────────────

class AgentExecutor:
    """
    Runs each AgentStep against the live broker, ML models, and data stores.
    All methods return plain dicts — never raise unless catastrophically broken.
    """

    PROFILE_FILE = Path("data/profile.json")
    TRADE_HISTORY = Path("data/trade_history.csv")

    def __init__(self, broker, strategy_manager, risk_manager, portfolio_manager, reasoning):
        self.broker    = broker
        self.sm        = strategy_manager
        self.rm        = risk_manager
        self.pm        = portfolio_manager
        self.reasoning = reasoning
        self.dynamic_targeter = DynamicRiskTargeter(self.broker)

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def run(self, step: AgentStep, plan: AgentPlan) -> dict:
        import time
        t0 = time.perf_counter()
        try:
            result = self._dispatch(step, plan)
        except Exception as exc:
            result = {"error": str(exc)}
        step.duration_ms = (time.perf_counter() - t0) * 1000
        return result or {}

    def _dynamic_risk_targets(self, symbol) -> dict:
        """Delegates structural target calculation to the standalone targeter."""
        return self.dynamic_targeter.calculate_targets(symbol)

    def _dispatch(self, step: AgentStep, plan: AgentPlan) -> dict:
        sym = plan.symbol
        handlers = {
            "market_regime":     lambda: self._market_regime(sym),
            "signal_ensemble":   lambda: self._signal_ensemble(sym),
            "quality_score":     lambda: self._quality_score(plan),
            "risk_check":        lambda: self._risk_check(sym),
            "position_sizing":   lambda: self._position_sizing(sym),
            "anomaly_check":     lambda: self._anomaly_check(sym),
            "correlation_hint":  lambda: self._correlation_hint(sym),
            "account_data":      lambda: self._account_data(),
            "open_positions":    lambda: self._open_positions(),
            "portfolio_health":  lambda: self._portfolio_health(),
            "regime_snapshot":   lambda: self._regime_snapshot(),
            "drawdown_analysis": lambda: self._drawdown_analysis(),
            "portfolio_scan":    lambda: self._portfolio_scan(),
            "best_opportunities":lambda: self._best_opportunities(plan),
            "dynamic_risk_targets": lambda: self._dynamic_risk_targets(sym),
            "risk_gate":         lambda: self._global_risk_gate(),
            "overtrading_check": lambda: self._overtrading_check(),
            "margin_check":      lambda: self._margin_check(),
            "floating_pnl":      lambda: self._floating_pnl(),
            "closed_trades":     lambda: self._closed_trades(),
            "performance_analysis":lambda: self._performance_analysis(plan),
            "risk_per_pos":      lambda: self._risk_per_pos(),
            "context_snapshot":  lambda: self._context_snapshot(),
            "live_price":        lambda: self._live_price(sym),
        }
        fn = handlers.get(step.name)
        return fn() if fn else {}

    # ── Step implementations ──────────────────────────────────────────────────

    def _market_regime(self, symbol) -> dict:
        try:
            learner = getattr(self.sm, "learner", None)
            if symbol and learner:
                raw = self.broker.ohclv_data(symbol, timeframe="H1", count=300)
                if raw is not None and not raw.empty:
                    from strategies.features.feature_engineer import FeatureEngineer
                    feat = FeatureEngineer.compute(raw)
                    if not feat.empty:
                        regime = learner.ingest_market_bar(feat.iloc[-1])
                        best   = learner.get_best_strategies_for_regime(regime)
                        insights = learner.generate_insights()
                        mults = {s: learner.get_confidence_multiplier(s, regime) for s in best}
                        return {
                            "regime": regime,
                            "best_strategies": best,
                            "confidence_multipliers": mults,
                            "insights": insights,
                        }
            regime = getattr(self.sm.learner, "_current_regime", "Unknown") if getattr(self.sm, "learner", None) else "Unknown"
            return {"regime": regime, "best_strategies": [], "insights": []}
        except Exception as e:
            return {"regime": "Unknown", "error": str(e)}

    def _signal_ensemble(self, symbol) -> dict:
        if not symbol:
            return {"action": "WAIT", "confidence": 0.0, "reason": "No symbol."}
        try:
            return self.sm.check_signals(symbol, use_ensemble=True)
        except Exception as e:
            return {"action": "WAIT", "confidence": 0.0, "reason": str(e)}

    def _quality_score(self, plan: AgentPlan) -> dict:
        sig_r    = self._get_step_result(plan, "signal_ensemble")
        regime_r = self._get_step_result(plan, "market_regime")
        htf_r = self._get_step_result(plan, "htf_trend_check")
        risk_r   = self._get_step_result(plan, "risk_check")
        
        if not sig_r:
            return {"grade": "D", "score": 0, "proceed": False, "notes": []}

        signal = {
            "action":          sig_r.get("action", "WAIT"),
            "confidence":      sig_r.get("confidence", 0.0),
            "reason":          sig_r.get("reason", ""),
            "source":          sig_r.get("source", ""),
            "lstm_prediction": sig_r.get("lstm_prediction", {}),
        }
        strategy_signals = sig_r.get("strategy_signals", {})
        regime   = regime_r.get("regime", "Unknown") if regime_r else "Unknown"
        risk_plan = risk_r.get("risk_plan", {"stop_loss_pips": 20, "take_profit_pips": 40}) 
        htf_trend = htf_r.get("trend", "NEUTRAL") if htf_r else "NEUTRAL"
        
        return self.reasoning.reasoner.score(
            signal, 
            strategy_signals,
            regime, 
            risk_plan, 
            self.reasoning,
            htf_trend
        )

    def _htf_trend_check(self, symbol) -> dict:
            """Determines the trend on a higher timeframe (e.g., Daily)."""
            if not symbol:
                return {"trend": "NEUTRAL"}
            
            try:
                # Fetch Daily (D1) candles to act as the higher timeframe
                df = self.broker.get_historical_rates(symbol, timeframe="D1", count=20)
                if df is None or df.empty:
                    return {"trend": "NEUTRAL"}
                
                # Simple SMA-20 Trend check
                closes = df['close'].values
                if len(closes) < 20:
                    return {"trend": "NEUTRAL"}
                    
                sma_20 = sum(closes[-20:]) / 20
                current_price = closes[-1]
                
                if current_price > sma_20:
                    trend = "BUY"
                elif current_price < sma_20:
                    trend = "SELL"
                else:
                    trend = "NEUTRAL"
                    
                return {"trend": trend}
            except Exception:
                return {"trend": "NEUTRAL"}

    def _risk_check(self, symbol) -> dict:
        cfg  = self._load_config()
        account = self.broker.getAccountInfo()

        allowed, reason = True, "No symbol."
        if symbol:
            portfolio_size = max(len(cfg.get("trading_symbols", [])), 1)
            allowed, reason = self.rm.is_trading_allowed(
                symbol, cfg.get("max_daily_loss", 500.0), portfolio_size
            )

        risk_comment = None
        if account:
            acct_dict = {
                "balance":      account.balance,
                "equity":       account.equity,
                "margin_level": account.margin_level,
            }
            risk_comment = self.reasoning.risk_commentator.comment(acct_dict, cfg)

        return {"allowed": allowed, "reason": reason, "risk_comment": risk_comment}

    def _position_sizing(self, symbol) -> dict:
        if not symbol:
            return {}
        cfg = self._load_config()
        try:
            return self.rm.calculate_safe_trade(
                symbol         = symbol,
                base_risk_pct  = cfg.get("risk_percentage", 1.0),
                stop_loss_pips = cfg.get("stop_loss", 20.0),
                max_daily_loss = cfg.get("max_daily_loss", 500.0),
                portfolio_size = max(len(cfg.get("trading_symbols", [])), 1),
            )
        except Exception as e:
            return {"approved": False, "reason": str(e)}

    def _anomaly_check(self, symbol) -> dict:
        try:
            learner = getattr(self.sm, "learner", None)
            if symbol and learner:
                raw = self.broker.ohclv_data(symbol, timeframe="H1", count=100)
                if raw is not None and not raw.empty:
                    from strategies.features.feature_engineer import FeatureEngineer
                    feat = FeatureEngineer.compute(raw)
                    if not feat.empty:
                        is_anom, score = learner.is_anomalous(feat.iloc[-1])
                        return {"is_anomaly": is_anom, "anomaly_score": round(score, 3)}
        except Exception:
            pass
        return {"is_anomaly": False, "anomaly_score": 0.0}

    def _correlation_hint(self, symbol) -> dict:
        cfg      = self._load_config()
        tracked  = cfg.get("trading_symbols", [])
        if not symbol or not tracked:
            return {"correlated": []}
        base = symbol[:3].upper()
        quote = symbol[3:].upper() if len(symbol) >= 6 else ""
        related = [
            s for s in tracked
            if s != symbol and (s[:3] == base or s[3:6] == quote or s[3:6] == base)
        ]
        return {"correlated": related[:3]}

    def _account_data(self) -> dict:
        account = self.broker.getAccountInfo()
        if not account:
            return {}
        return {
            "balance":      round(account.balance, 2),
            "equity":       round(account.equity, 2),
            "profit":       round(account.profit, 2),
            "margin_level": round(account.margin_level, 2) if account.margin_level else None,
            "margin":       round(getattr(account, "margin", 0.0), 2),
            "free_margin":  round(getattr(account, "margin_free", 0.0), 2),
        }

    def _open_positions(self) -> dict:
        positions = self.broker.getPositions()
        if not positions:
            return {"count": 0, "positions": [], "total_pnl": 0.0}
        pos_list = []
        total = 0.0
        for p in positions:
            pos_list.append({
                "symbol":     p.symbol,
                "action":     "BUY" if p.type == 0 else "SELL",
                "volume":     p.volume,
                "price_open": p.price_open,
                "profit":     round(p.profit, 2),
                "ticket":     p.ticket,
            })
            total += p.profit
        return {"count": len(pos_list), "positions": pos_list, "total_pnl": round(total, 2)}

    def _portfolio_health(self) -> dict:
        try:
            health = self.pm.get_portfolio_health()
            return {"health_summary": health}
        except Exception:
            return {"health_summary": "Unable to assess."}

    def _regime_snapshot(self) -> dict:
        cfg     = self._load_config()
        learner = getattr(self.sm, "learner", None)
        regime  = learner.get_current_regime() if learner else "Unknown"
        best    = learner.get_best_strategies_for_regime(regime) if learner else []
        return {
            "regime":           regime,
            "best_strategies":  best,
            "tracked_symbols":  cfg.get("trading_symbols", []),
        }

    def _drawdown_analysis(self) -> dict:
        account = self.broker.getAccountInfo()
        if not account:
            return {}
        balance = account.balance
        equity  = account.equity
        high    = getattr(self.rm, "daily_high_watermark", balance)
        low     = getattr(self.rm, "daily_low_watermark", equity)
        trailing = max(high - equity, 0.0)
        dd_pct   = (trailing / high * 100) if high > 0 else 0.0
        return {
            "balance":          round(balance, 2),
            "equity":           round(equity, 2),
            "daily_high":       round(high, 2),
            "daily_low":        round(low, 2),
            "trailing_drawdown":round(trailing, 2),
            "drawdown_pct":     round(dd_pct, 2),
        }

    def _portfolio_scan(self) -> dict:
        cfg = self._load_config()
        try:
            results = self.pm.evaluate_portfolio_opportunities(
                risk_pct      = cfg.get("risk_percentage", 1.0),
                stop_loss     = cfg.get("stop_loss", 20.0),
                max_daily_loss= cfg.get("max_daily_loss", 500.0),
            )
            return {"scan_results": results}
        except Exception as e:
            return {"scan_results": [f"Scan error: {e}"]}

    def _best_opportunities(self, plan: AgentPlan) -> dict:
        scan_r = self._get_step_result(plan, "portfolio_scan") or {}
        results = scan_r.get("scan_results", [])
        executed = [r for r in results if "🟢" in r or "EXECUTED" in r]
        blocked  = [r for r in results if "⚠️" in r or "❌" in r or "FAILED" in r]
        closed   = [r for r in results if "MARKET CLOSED" in r]
        waiting  = [r for r in results if "WAIT" in r.upper() or "No high" in r]
        return {
            "executed": executed,
            "blocked":  blocked,
            "closed":   closed,
            "waiting":  waiting,
            "total":    len(results),
        }

    def _global_risk_gate(self) -> dict:
        account = self.broker.getAccountInfo()
        if not account:
            return {"gate_open": False, "reason": "Cannot fetch account data"}
        cfg      = self._load_config()
        max_loss = cfg.get("max_daily_loss", 500.0)
        high     = getattr(self.rm, "daily_high_watermark", account.balance)
        trailing = max(high - account.equity, 0.0)
        gate_open = trailing < max_loss
        return {
            "gate_open": gate_open,
            "reason":    "OK" if gate_open else f"Drawdown ${trailing:.2f} ≥ limit ${max_loss:.2f}",
            "drawdown":  round(trailing, 2),
            "balance":   round(account.balance, 2),
            "equity":    round(account.equity, 2),
        }

    def _overtrading_check(self) -> dict:
        try:
            if not self.TRADE_HISTORY.exists():
                return {"warning": None, "recent_count": 0}
            trades = []
            with open(self.TRADE_HISTORY) as f:
                for row in csv.DictReader(f):
                    trades.append(row)
            recent  = trades[-30:]
            warning = self.reasoning.risk_commentator.overtrading_check(recent)
            return {"warning": warning, "recent_count": len(recent), "total": len(trades)}
        except Exception:
            return {"warning": None, "recent_count": 0}

    def _margin_check(self) -> dict:
        account = self.broker.getAccountInfo()
        if not account:
            return {}
        return {
            "margin_level": account.margin_level,
            "margin":       getattr(account, "margin", 0.0),
            "free_margin":  getattr(account, "margin_free", 0.0),
        }

    def _floating_pnl(self) -> dict:
        positions = self.broker.getPositions()
        if not positions:
            return {"total": 0.0, "winning": 0.0, "losing": 0.0, "count": 0}
        winning = sum(p.profit for p in positions if p.profit > 0)
        losing  = sum(p.profit for p in positions if p.profit <= 0)
        return {
            "total":   round(winning + losing, 2),
            "winning": round(winning, 2),
            "losing":  round(losing, 2),
            "count":   len(positions),
        }

    def _closed_trades(self) -> dict:
        try:
            if not self.TRADE_HISTORY.exists():
                return {"trades": [], "pnl": 0.0, "count": 0}
            today   = date.today().strftime("%Y-%m-%d")
            trades  = []
            pnl     = 0.0
            buys = sells = closes = 0
            with open(self.TRADE_HISTORY) as f:
                for row in csv.DictReader(f):
                    if row.get("Timestamp", "").startswith(today):
                        trades.append(row)
                        action = row.get("Action", "")
                        if action == "BUY":   buys   += 1
                        elif action == "SELL": sells  += 1
                        elif action == "CLOSE":
                            closes += 1
                            try:
                                pnl += float(row.get("Comment", "0").replace("Profit:", "").strip())
                            except ValueError:
                                pass
            return {
                "trades": trades, "pnl": round(pnl, 2),
                "count": len(trades), "buys": buys, "sells": sells, "closes": closes,
            }
        except Exception:
            return {"trades": [], "pnl": 0.0, "count": 0}

    def _performance_analysis(self, plan: AgentPlan) -> dict:
        hist_r = self._get_step_result(plan, "closed_trades") or {}
        trades = hist_r.get("trades", [])
        by_sym: dict[str, float] = {}
        for t in trades:
            if t.get("Action") == "CLOSE":
                sym = t.get("Symbol", "")
                try:
                    val = float(t.get("Comment", "0").replace("Profit:", "").strip())
                    by_sym[sym] = by_sym.get(sym, 0.0) + val
                except ValueError:
                    pass
        winning = [k for k, v in by_sym.items() if v > 0]
        losing  = [k for k, v in by_sym.items() if v <= 0]

        # Strategy breakdown
        strat_counts: dict[str, int] = {}
        for t in trades:
            s = t.get("Strategy", "Unknown")
            strat_counts[s] = strat_counts.get(s, 0) + 1

        return {
            "by_symbol":        by_sym,
            "winning_symbols":  winning,
            "losing_symbols":   losing,
            "best_performer":   max(by_sym, key=by_sym.get) if by_sym else None,
            "worst_performer":  min(by_sym, key=by_sym.get) if by_sym else None,
            "strategy_counts":  strat_counts,
        }

    def _risk_per_pos(self) -> dict:
        positions = self.broker.getPositions()
        account   = self.broker.getAccountInfo()
        if not positions or not account:
            return {"items": [], "total_exposure": 0}
        items = []
        for p in positions:
            risk_pct = abs(p.profit) / account.equity * 100 if account.equity > 0 else 0
            items.append({
                "symbol":   p.symbol,
                "profit":   round(p.profit, 2),
                "risk_pct": round(risk_pct, 2),
                "volume":   p.volume,
                "action":   "BUY" if p.type == 0 else "SELL",
            })
        return {"items": items, "total_exposure": len(items)}

    def _context_snapshot(self) -> dict:
        account   = self.broker.getAccountInfo()
        positions = self.broker.getPositions()
        learner   = getattr(self.sm, "learner", None)
        regime    = learner.get_current_regime() if learner else "Unknown"
        return {
            "balance":        round(account.balance, 2) if account else 0,
            "equity":         round(account.equity, 2)  if account else 0,
            "open_positions": len(positions) if positions else 0,
            "regime":         regime,
        }

    def _live_price(self, symbol) -> dict:
        if not symbol:
            return {}
        try:
            tick = self.broker.get_tick_data(symbol)
            if tick:
                return {
                    "symbol": symbol,
                    "bid":    tick.get("bid", 0),
                    "ask":    tick.get("ask", 0),
                    "spread": round((tick.get("ask", 0) - tick.get("bid", 0)) * 10000, 1),
                }
        except Exception:
            pass
        return {"symbol": symbol}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            return json.loads(self.PROFILE_FILE.read_text())
        except Exception:
            return {
                "risk_percentage": 1.0, "stop_loss": 20.0,
                "max_daily_loss": 500.0, "trading_symbols": ["EURUSD"],
            }

    @staticmethod
    def _get_step_result(plan: AgentPlan, name: str) -> Optional[dict]:
        for step in plan.steps:
            if step.name == name and step.result:
                return step.result
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AgentSynthesizer
# ─────────────────────────────────────────────────────────────────────────────

class AgentSynthesizer:
    """
    Deterministic Natural Language Generation (NLG) engine.
    Creates dynamic, non-repetitive English responses from raw trading data
    without relying on external LLMs.
    """
    def __init__(self, reasoning):
        self.reasoning = reasoning
        
        # ── Lexical Dictionary ──
        self.lexicon = {
            "greetings": [
                "System online.", "Ready.", "Awaiting input.", "All systems green."
            ],
            "buy_openers": [
                "I'm seeing bullish confluence on {symbol}.",
                "The structure on {symbol} is flagging a long entry.",
                "Buying pressure is building on {symbol}.",
                "We have a clear upward bias on {symbol}."
            ],
            "sell_openers": [
                "Bearish distribution detected on {symbol}.",
                "The structure on {symbol} is rolling over short.",
                "Sell signals are aligning for {symbol}.",
                "We have a clear downward bias on {symbol}."
            ],
            "wait_openers": [
                "{symbol} is currently a mixed bag.",
                "I have no clear statistical edge on {symbol} right now.",
                "Conditions on {symbol} are too choppy to call.",
                "Staying flat on {symbol}."
            ],
            "transitions_add": [
                "Furthermore,", "Additionally,", "To add to that,", "Also,"
            ],
            "transitions_contrast": [
                "However,", "That said,", "On the flip side,", "Despite this,"
            ],
            "grades": {
                "A": ["This is a premium, high-probability setup.", "Conditions are optimal.", "I highly recommend this trade."],
                "B": ["It's a solid setup with good alignment.", "There is a viable edge here.", "Favorable risk profile overall."],
                "C": ["Proceed with caution.", "It's a marginal setup at best.", "You might want to wait for cleaner price action."],
                "D": ["I strongly advise sitting this one out.", "The math does not support a trade here.", "Too much conflicting data."]
            }
        }

    def synthesize(self, plan: 'AgentPlan') -> str:
        """Main routing function for synthesis."""
        if plan.intent in ["analyze_symbol", "ai_analysis"]:
            return self._synth_dynamic_analysis(plan)
        elif plan.intent == "greeting":
            return random.choice(self.lexicon["greetings"])
        
        # Fallback for simple actions
        return f"Action '{plan.intent}' processed successfully."

    def _synth_dynamic_analysis(self, plan: 'AgentPlan') -> str:
        """Builds a flowing paragraph dynamically from the execution notes."""
        symbol = plan.symbol or "the asset"
        r = plan.context  # The dictionary populated by the executor
        
        grade = r.get("grade", "D")
        direction = r.get("action", "WAIT")
        notes = r.get("notes", [])
        
        # 1. Choose an opener based on direction
        if direction == "BUY":
            opener = random.choice(self.lexicon["buy_openers"]).format(symbol=symbol)
        elif direction == "SELL":
            opener = random.choice(self.lexicon["sell_openers"]).format(symbol=symbol)
        else:
            opener = random.choice(self.lexicon["wait_openers"]).format(symbol=symbol)

        # 2. Process the notes (Extracting the English reasoning from the points)
        # Assuming notes look like: "HTF Trend Aligned (BUY) → +15 pts"
        clean_notes = []
        for note in notes:
            # Strip the points part (e.g., " → +15 pts") to make it conversational
            clean_text = note.split(" → ")[0]
            clean_notes.append(clean_text)

        # 3. Build the body paragraph
        body_sentences = []
        if clean_notes:
            body_sentences.append(f"Specifically, {clean_notes[0].lower()}.")
            
            if len(clean_notes) > 1:
                trans = random.choice(self.lexicon["transitions_add"])
                body_sentences.append(f"{trans} {clean_notes[1].lower()}.")
                
            if len(clean_notes) > 2:
                # Group the remaining notes into a list
                remaining = ", and ".join([n.lower() for n in clean_notes[2:4]])
                body_sentences.append(f"We are also seeing that {remaining}.")

        body_text = " ".join(body_sentences)

        # 4. Append the Grade / Verdict
        verdict = random.choice(self.lexicon["grades"].get(grade, self.lexicon["grades"]["D"]))
        
        # 5. Final Assembly
        final_response = f"{opener} {body_text} {verdict} (Grade: {grade})"
        
        return final_response
# ─────────────────────────────────────────────────────────────────────────────
# AgentCore  (public façade)
# ─────────────────────────────────────────────────────────────────────────────

class AgentCore:
    """
    Façade that wires AgentPlanner → AgentExecutor → AgentSynthesizer.

    Usage
    -----
        core = AgentCore(strategy_manager, risk_manager,
                         portfolio_manager, broker, reasoning_engine)

        response = core.run(intent="analyze_symbol",
                            entities={"symbols": ["EURUSD"]},
                            memory={"last_symbol": "EURUSD"},
                            step_callback=print)

        # plan.suggested_action is set if the agent recommends an action
        plan = core.last_plan
    """

    def __init__(self, strategy_manager, risk_manager, portfolio_manager, broker, reasoning_engine):
        self.planner     = AgentPlanner()
        self.executor    = AgentExecutor(broker, strategy_manager, risk_manager,
                                         portfolio_manager, reasoning_engine)
        self.synthesizer = AgentSynthesizer(reasoning_engine)
        self.last_plan: Optional[AgentPlan] = None

    def run(
        self,
        intent:         str,
        entities:       dict,
        memory:         dict,
        step_callback=None,    # optional fn(step_name, description) for live progress
    ) -> str:
        # 1. Plan
        plan = self.planner.plan(intent, entities, memory)

        # 2. Execute
        for step in plan.steps:
            step.status = "running"
            if step_callback:
                step_callback(step.name, step.description)
            step.result = self.executor.run(step, plan)
            step.status = "done" if "error" not in (step.result or {}) else "failed"

        risk_step = plan.context.get("dynamic_risk_targets")
        if risk_step:
            plan.context["dynamic_targets"] = risk_step

        response = self.synthesizer.synthesize(plan)
        plan.final_response = response
        self.last_plan = plan

        return response