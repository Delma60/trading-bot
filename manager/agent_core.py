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
        risk_plan = {"stop_loss_pips": 20, "take_profit_pips": 40}

        return self.reasoning.reasoner.score(
            signal, strategy_signals, regime, risk_plan, self.reasoning
        )

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
    Merges AgentStep results into a single, coherent, data-rich response.
    Every factual claim references a live number from the step results.
    """

    def __init__(self, reasoning):
        self.reasoning = reasoning

    def synthesize(self, plan: AgentPlan) -> str:
        r = {s.name: (s.result or {}) for s in plan.steps}
        intent = plan.intent

        dispatch = {
            "analyze_symbol":   self._synth_analysis,
            "execute_trade":    self._synth_trade,
            "portfolio_status": self._synth_portfolio,
            "bulk_scan":        self._synth_scan,
            "risk_management":  self._synth_risk,
            "account_summary":  self._synth_account,
            "trade_history":    self._synth_history,
            "active_positions": self._synth_positions,
            "get_price":        self._synth_price,
            "close_position":   self._synth_close,
            "close_all":        self._synth_close_all,
            "news_update":      self._synth_news,
            "greeting":         self._synth_greeting,
            "strategy_info":    self._synth_strategy,
            "general":          self._synth_general,
        }
        fn = dispatch.get(intent, self._synth_general)
        return fn(plan, r)

    # ── Intent synthesizers ───────────────────────────────────────────────────

    def _synth_analysis(self, plan: AgentPlan, r: dict) -> str:
        sym      = plan.symbol or "UNKNOWN"
        sig_r    = r.get("signal_ensemble", {})
        regime_r = r.get("market_regime", {})
        score_r  = r.get("quality_score", {})
        risk_r   = r.get("risk_check", {})
        size_r   = r.get("position_sizing", {})
        anom_r   = r.get("anomaly_check", {})
        corr_r   = r.get("correlation_hint", {})

        action     = sig_r.get("action", "WAIT")
        confidence = sig_r.get("confidence", 0.0)
        source     = sig_r.get("source", "ensemble")
        lstm       = sig_r.get("lstm_prediction", {})
        strat_sigs = sig_r.get("strategy_signals", {})

        regime    = regime_r.get("regime", "Unknown")
        best_s    = regime_r.get("best_strategies", [])
        grade     = score_r.get("grade", "D")
        score     = score_r.get("score", 0)
        proceed   = score_r.get("proceed", False)
        notes     = score_r.get("notes", [])

        risk_ok   = risk_r.get("allowed", True)
        risk_msg  = risk_r.get("reason", "")
        risk_cmt  = risk_r.get("risk_comment")

        lots      = size_r.get("lots", 0.0)
        sl_pips   = size_r.get("stop_loss_pips", 20.0)
        risk_usd  = size_r.get("risk_usd", 0.0)

        is_anom   = anom_r.get("is_anomaly", False)
        anom_score= anom_r.get("anomaly_score", 0.0)
        corr      = corr_r.get("correlated", [])

        parts = []

        # ── 1. Signal narration ───────────────────────────────────────────────
        sig_dict = {
            "action": action, "confidence": confidence,
            "reason": sig_r.get("reason", ""),
            "source": source, "lstm_prediction": lstm,
        }
        parts.append(self.reasoning.narrator.narrate(sig_dict, sym))

        # ── 2. Regime + best strategies ───────────────────────────────────────
        regime_line = self.reasoning.regime_advisor.advise(regime, self.reasoning)
        if best_s:
            fit = ", ".join(best_s[:2])
            parts.append(f"Regime: '{regime}'. Best-fit strategies right now: {fit}. {regime_line}")
        else:
            parts.append(f"Regime: '{regime}'. {regime_line}")

        # ── 3. Strategy consensus ─────────────────────────────────────────────
        if strat_sigs:
            agree  = sum(1 for s in strat_sigs.values() if s.get("action") == action and s.get("confidence", 0) > 0.3)
            total  = len(strat_sigs)
            consensus = f"{agree}/{total} strategies agree."
            lstm_dir  = lstm.get("direction", "NEUTRAL")
            lstm_conf = lstm.get("confidence", 0.0)
            align = "confirms" if (
                (action == "BUY" and lstm_dir == "UP") or
                (action == "SELL" and lstm_dir == "DOWN")
            ) else "conflicts with" if lstm_dir != "NEUTRAL" else "is neutral on"
            parts.append(f"{consensus} LSTM {align} the signal ({lstm_dir} @ {lstm_conf:.0%}).")

        # ── 4. Quality grade ──────────────────────────────────────────────────
        grade_text = {
            "A": f"Quality grade A ({score}/100) — strong confluence, high-probability entry.",
            "B": f"Quality grade B ({score}/100) — reasonable setup, manage size carefully.",
            "C": f"Quality grade C ({score}/100) — weak confluence. Consider waiting.",
            "D": f"Quality grade D ({score}/100) — poor setup. I'd pass on this one.",
        }.get(grade, f"Grade {grade} ({score}/100).")
        parts.append(grade_text)

        # ── 5. Anomaly warning ────────────────────────────────────────────────
        if is_anom:
            parts.append(
                f"⚠️ Anomaly detected (score {anom_score:.2f}) — market conditions are unusual. "
                f"Reduce size or wait for conditions to normalise."
            )

        # ── 6. Risk gate ──────────────────────────────────────────────────────
        if not risk_ok:
            parts.append(f"🛑 Risk gate BLOCKED: {risk_msg}")
        elif risk_cmt:
            parts.append(f"💡 {risk_cmt}")

        # ── 7. Correlated pairs ───────────────────────────────────────────────
        if corr:
            parts.append(f"Correlated pairs worth watching: {', '.join(corr)}.")

        # ── 8. Actionable recommendation ─────────────────────────────────────
        if proceed and risk_ok and not is_anom and action != "WAIT":
            if lots and lots > 0:
                parts.append(
                    f"📊 Suggested entry: {action} {lots} lots | SL: {sl_pips:.0f} pips | "
                    f"Risk: ~${risk_usd:.2f}. "
                    f"Say 'execute' or 'place the trade' to proceed."
                )
                plan.suggested_action = f"{action} {lots} {sym}"
            else:
                parts.append(
                    f"📊 Signal: {action} on {sym}. Position sizing failed — check margin. "
                    f"Consider micro-lots (0.01)."
                )
        elif action == "WAIT":
            parts.append("Staying flat — no clean entry yet.")

        return " ".join(parts)

    def _synth_trade(self, plan: AgentPlan, r: dict) -> str:
        # For trade intents, output the same analysis as analyze_symbol;
        # the actual execution is handled by chat.py's action executor.
        return self._synth_analysis(plan, r)

    def _synth_portfolio(self, plan: AgentPlan, r: dict) -> str:
        acct_r    = r.get("account_data", {})
        pos_r     = r.get("open_positions", {})
        health_r  = r.get("portfolio_health", {})
        regime_r  = r.get("regime_snapshot", {})
        dd_r      = r.get("drawdown_analysis", {})

        balance   = acct_r.get("balance", 0)
        equity    = acct_r.get("equity", 0)
        ml        = acct_r.get("margin_level")
        pos_count = pos_r.get("count", 0)
        total_pnl = pos_r.get("total_pnl", 0.0)
        health    = health_r.get("health_summary", "")
        regime    = regime_r.get("regime", "Unknown")
        best_s    = regime_r.get("best_strategies", [])
        trailing  = dd_r.get("trailing_drawdown", 0.0)
        dd_pct    = dd_r.get("drawdown_pct", 0.0)

        lines = [
            f"Account — Balance: ${balance:,.2f} | Equity: ${equity:,.2f} | "
            f"Floating P&L: ${total_pnl:+.2f}.",
        ]

        if ml:
            ml_status = "✅" if ml > 300 else "⚠️" if ml > 150 else "🔴"
            lines.append(f"Margin level: {ml_status} {ml:.0f}%.")

        if pos_count:
            positions = pos_r.get("positions", [])
            pos_strs  = [
                f"{'🟢' if p['profit']>=0 else '🔴'} {p['symbol']} "
                f"{p['action']} {p['volume']}L (${p['profit']:+.2f})"
                for p in positions[:5]
            ]
            lines.append(f"{pos_count} open position(s): " + " | ".join(pos_strs))
        else:
            lines.append("No open positions — portfolio is flat.")

        lines.append(f"Daily trailing drawdown: ${trailing:.2f} ({dd_pct:.1f}%).")

        if regime != "Unknown":
            strat_hint = f" Best fit: {', '.join(best_s[:2])}." if best_s else ""
            lines.append(f"Market regime: '{regime}'.{strat_hint}")

        if health:
            lines.append(health)

        return "\n".join(lines)

    def _synth_scan(self, plan: AgentPlan, r: dict) -> str:
        gate_r    = r.get("risk_gate", {})
        best_r    = r.get("best_opportunities", {})
        regime_r  = r.get("regime_snapshot", {})

        if not gate_r.get("gate_open", True):
            dd = gate_r.get("drawdown", 0)
            return (
                f"🛑 Risk gate CLOSED — trailing drawdown ${dd:.2f} has hit the daily limit. "
                f"No new trades until tomorrow or until the gate reopens."
            )

        executed = best_r.get("executed", [])
        blocked  = best_r.get("blocked", [])
        closed   = best_r.get("closed", [])
        regime   = regime_r.get("regime", "Unknown")
        best_s   = regime_r.get("best_strategies", [])

        parts = [f"Scan complete. Regime: '{regime}'."]

        if executed:
            parts.append(f"✅ {len(executed)} trade(s) executed:")
            parts.extend(f"  {e}" for e in executed[:5])
        else:
            parts.append("No signals met the execution threshold this cycle.")

        if blocked:
            parts.append(f"⚠️ {len(blocked)} symbol(s) blocked by risk rules.")

        if closed:
            parts.append(f"⏳ {len(closed)} market(s) currently closed.")

        if best_s:
            parts.append(f"Best-fit strategies for current conditions: {', '.join(best_s[:2])}.")

        return "\n".join(parts)

    def _synth_risk(self, plan: AgentPlan, r: dict) -> str:
        acct_r  = r.get("account_data", {})
        dd_r    = r.get("drawdown_analysis", {})
        ot_r    = r.get("overtrading_check", {})
        mg_r    = r.get("margin_check", {})
        pos_r   = r.get("open_positions", {})

        balance  = acct_r.get("balance", 0)
        equity   = acct_r.get("equity", 0)
        high     = dd_r.get("daily_high", balance)
        trailing = dd_r.get("trailing_drawdown", 0.0)
        dd_pct   = dd_r.get("drawdown_pct", 0.0)
        ml       = mg_r.get("margin_level") or acct_r.get("margin_level")
        pos_cnt  = pos_r.get("count", 0)

        lines = [
            f"Risk Snapshot — Balance: ${balance:,.2f} | Equity: ${equity:,.2f}",
            f"Daily peak: ${high:,.2f} | Trailing drawdown: ${trailing:.2f} ({dd_pct:.1f}%)",
        ]

        if ml:
            status = "✅ Healthy" if ml > 300 else "⚠️ Elevated" if ml > 150 else "🔴 Critical"
            lines.append(f"Margin level: {ml:.0f}% — {status}.")

        lines.append(f"Open positions: {pos_cnt}.")

        ot_warning = ot_r.get("warning")
        if ot_warning:
            lines.append(ot_warning)
        else:
            lines.append("No overtrading patterns detected.")

        return "\n".join(lines)

    def _synth_account(self, plan: AgentPlan, r: dict) -> str:
        acct_r   = r.get("account_data", {})
        float_r  = r.get("floating_pnl", {})
        regime_r = r.get("regime_snapshot", {})

        balance  = acct_r.get("balance", 0)
        equity   = acct_r.get("equity", 0)
        ml       = acct_r.get("margin_level")
        floating = float_r.get("total", 0.0)
        win_pnl  = float_r.get("winning", 0.0)
        lose_pnl = float_r.get("losing", 0.0)
        n_pos    = float_r.get("count", 0)
        regime   = regime_r.get("regime", "Unknown")

        lines = [
            f"Balance: ${balance:,.2f} | Equity: ${equity:,.2f}",
            f"Floating P&L: ${floating:+.2f} "
            f"(Winners +${win_pnl:.2f} | Losers ${lose_pnl:.2f}) across {n_pos} position(s).",
        ]
        if ml:
            lines.append(f"Margin level: {ml:.0f}%.")
        lines.append(f"Market regime: '{regime}'.")
        return " ".join(lines)

    def _synth_history(self, plan: AgentPlan, r: dict) -> str:
        hist_r  = r.get("closed_trades", {})
        perf_r  = r.get("performance_analysis", {})
        ot_r    = r.get("overtrading_check", {})

        count   = hist_r.get("count", 0)
        pnl     = hist_r.get("pnl", 0.0)
        buys    = hist_r.get("buys", 0)
        sells   = hist_r.get("sells", 0)
        closes  = hist_r.get("closes", 0)

        by_sym  = perf_r.get("by_symbol", {})
        best    = perf_r.get("best_performer")
        worst   = perf_r.get("worst_performer")
        strat_c = perf_r.get("strategy_counts", {})

        if count == 0:
            return "No trades logged today yet. The slate is clean."

        pnl_str = f"${abs(pnl):.2f} {'profit' if pnl >= 0 else 'loss'}"
        lines   = [
            f"Today — {count} trades ({buys} BUY, {sells} SELL, {closes} closed) | P&L: {pnl_str}."
        ]

        if best and by_sym:
            lines.append(f"Best: {best} (${by_sym[best]:+.2f}).")
        if worst and by_sym and worst != best:
            lines.append(f"Worst: {worst} (${by_sym[worst]:+.2f}).")

        if strat_c:
            top_strat = max(strat_c, key=strat_c.get)
            lines.append(f"Most active strategy: {top_strat} ({strat_c[top_strat]} trades).")

        ot_w = ot_r.get("warning")
        if ot_w:
            lines.append(ot_w)

        return " ".join(lines)

    def _synth_positions(self, plan: AgentPlan, r: dict) -> str:
        pos_r    = r.get("open_positions", {})
        risk_r   = r.get("risk_per_pos", {})
        health_r = r.get("portfolio_health", {})

        count    = pos_r.get("count", 0)
        positions= pos_r.get("positions", [])
        total    = pos_r.get("total_pnl", 0.0)
        health   = health_r.get("health_summary", "")

        if count == 0:
            return "No open positions. Portfolio is flat — good time to scan for entries."

        lines = [f"{count} open position(s) | Floating P&L: ${total:+.2f}"]
        for p in positions:
            icon = "🟢" if p["profit"] >= 0 else "🔴"
            lines.append(
                f"  {icon} {p['symbol']} {p['action']} {p['volume']}L "
                f"@ {p['price_open']} | P&L ${p['profit']:+.2f} | #{p['ticket']}"
            )

        high_risk = [x for x in risk_r.get("items", []) if x["risk_pct"] > 5]
        if high_risk:
            lines.append(f"⚠️ High exposure: {', '.join(x['symbol'] for x in high_risk)}")

        if health:
            lines.append(health)

        return "\n".join(lines)

    def _synth_price(self, plan: AgentPlan, r: dict) -> str:
        price_r  = r.get("live_price", {})
        regime_r = r.get("market_regime", {})
        sym      = price_r.get("symbol", plan.symbol or "?")
        bid      = price_r.get("bid", 0)
        ask      = price_r.get("ask", 0)
        spread   = price_r.get("spread", 0)
        regime   = regime_r.get("regime", "Unknown")

        if not bid:
            return f"Could not fetch live price for {sym}. Check if market is open."

        return (
            f"{sym} — Bid: {bid} | Ask: {ask} | Spread: {spread:.1f} pips. "
            f"Regime: '{regime}'."
        )

    def _synth_close(self, plan: AgentPlan, r: dict) -> str:
        pos_r = r.get("open_positions", {})
        sym   = plan.symbol
        positions = [p for p in pos_r.get("positions", []) if p["symbol"] == sym]
        if not positions:
            return f"No open positions found for {sym}."
        plan.suggested_action = f"CLOSE {sym}"
        return (
            f"Found {len(positions)} open position(s) for {sym}. "
            f"Confirm close? Say 'yes close {sym}' to proceed."
        )

    def _synth_close_all(self, plan: AgentPlan, r: dict) -> str:
        pos_r   = r.get("open_positions", {})
        float_r = r.get("floating_pnl", {})
        count   = pos_r.get("count", 0)
        total   = float_r.get("total", 0.0)
        if count == 0:
            return "No open positions to close."
        plan.suggested_action = "CLOSE_ALL"
        return (
            f"You have {count} open position(s) with ${total:+.2f} floating P&L. "
            f"Confirm to close all? Say 'yes close all' to proceed."
        )

    def _synth_news(self, plan: AgentPlan, r: dict) -> str:
        regime_r = r.get("regime_snapshot", {})
        acct_r   = r.get("account_data", {})
        regime   = regime_r.get("regime", "Unknown")
        best_s   = regime_r.get("best_strategies", [])
        equity   = acct_r.get("equity", 0)
        strat_hint = f" Best-fit strategies: {', '.join(best_s[:2])}." if best_s else ""

        return (
            f"I don't have a live news feed, but here's the current picture: "
            f"market regime is '{regime}'.{strat_hint} "
            f"Account equity: ${equity:,.2f}. "
            f"Check ForexFactory or Bloomberg for scheduled events."
        )

    def _synth_greeting(self, plan: AgentPlan, r: dict) -> str:
        acct_r   = r.get("account_data", {})
        pos_r    = r.get("open_positions", {})
        regime_r = r.get("regime_snapshot", {})

        balance  = acct_r.get("balance", 0)
        equity   = acct_r.get("equity", 0)
        pos_cnt  = pos_r.get("count", 0)
        total_pnl= pos_r.get("total_pnl", 0.0)
        regime   = regime_r.get("regime", "Unknown")
        best_s   = regime_r.get("best_strategies", [])

        account_line = (
            f"Balance ${balance:,.2f} | Equity ${equity:,.2f}."
            if balance else "Broker connected."
        )

        if pos_cnt:
            pos_line = f"{pos_cnt} trade(s) open, floating ${total_pnl:+.2f}."
        else:
            pos_line = "Portfolio is flat — no open positions."

        if regime != "Unknown" and best_s:
            regime_line = f"Market is in a '{regime}' regime. Best fit: {', '.join(best_s[:2])}."
        else:
            regime_line = ""

        suggestions = [
            "Run a portfolio scan?",
            "Analyse a symbol?",
            "Check open positions?",
            "Review risk metrics?",
        ]
        opener = random.choice([
            "ARIA online.", "Back online.", "Systems live.", "Ready.",
        ])
        return f"{opener} {account_line} {pos_line} {regime_line} {random.choice(suggestions)}"

    def _synth_strategy(self, plan: AgentPlan, r: dict) -> str:
        regime_r = r.get("regime_snapshot", {})
        acct_r   = r.get("account_data", {})
        regime   = regime_r.get("regime", "Unknown")
        best_s   = regime_r.get("best_strategies", [])
        tracked  = regime_r.get("tracked_symbols", [])

        lines = [
            f"Current regime: '{regime}'.",
            f"Best-fit strategies for this environment: {', '.join(best_s) if best_s else 'Not determined yet.'}",
            f"Tracking {len(tracked)} symbol(s): {', '.join(tracked[:5])}{'...' if len(tracked) > 5 else ''}.",
            "Ensemble pipeline: FeatureEngineer → 7 strategy engines → LSTM predictor → XGBoost meta-scorer.",
            "Signal threshold: >30% confidence to suggest, >75% to auto-execute in scan mode.",
        ]
        return "\n".join(lines)

    def _synth_general(self, plan: AgentPlan, r: dict) -> str:
        ctx = r.get("context_snapshot", {})
        balance  = ctx.get("balance", 0)
        n_pos    = ctx.get("open_positions", 0)
        regime   = ctx.get("regime", "Unknown")
        suggestions = [
            "scan the portfolio", "analyse a symbol", "check positions",
            "review your risk metrics", "check account balance",
        ]
        return (
            f"Not sure what you meant. "
            f"Context: Balance ${balance:,.2f} | {n_pos} open position(s) | Regime '{regime}'. "
            f"Try: {random.choice(suggestions)}."
        )


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

        # 3. Synthesize
        response = self.synthesizer.synthesize(plan)
        plan.final_response = response
        self.last_plan = plan

        return response