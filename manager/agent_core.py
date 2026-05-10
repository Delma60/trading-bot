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
from manager.profile_manager import profile
import pandas as pd
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
        symbol = symbols[0] if symbols else memory.get("last_symbol", profile.symbols()[0] if profile.symbols() else None)
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
            AgentStep("htf_trend_check",  f"Higher-timeframe trend — {symbol}"),
            AgentStep("signal_ensemble",  f"Running strategy ensemble on {symbol}"),
            AgentStep("quality_score",    "Scoring signal quality"),
            AgentStep("risk_check",       "Risk gate validation"),
            AgentStep("position_sizing",  "Calculating optimal lot size"),
            AgentStep("anomaly_check",    "Checking for anomalous conditions"),
            AgentStep("correlation_hint", "Scanning correlated pairs"),
        ])

    def _plan_trade(self, symbol, entities, memory) -> AgentPlan:
        return AgentPlan(intent="execute_trade", symbol=symbol, steps=[
            AgentStep("market_regime",        f"Market regime check — {symbol}"),
            AgentStep("htf_trend_check",      f"Higher-timeframe trend — {symbol}"),
            AgentStep("signal_ensemble",      f"Signal validation — {symbol}"),
            AgentStep("quality_score",        "Signal quality scoring"),
            AgentStep("dynamic_risk_targets", "Calculating structural SL/TP targets"),
            AgentStep("risk_check",           "Risk gate validation"),
            AgentStep("position_sizing",      "Position sizing calculation"),
            AgentStep("anomaly_check",        "Anomaly detection"),
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
        return AgentPlan(intent="news_update", symbol=symbol, steps=[
            AgentStep("fetch_news",      "Scanning latest market intelligence..."),
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

    # PROFILE_FILE = Path("data/profile.json")
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
            "htf_trend_check": lambda: self._htf_trend_check(sym),
            "fetch_news":        lambda: self._fetch_news(sym),
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
        if not self.reasoning or not hasattr(self.reasoning, "reasoner"):
            return {"grade": "D", "score": 0, "proceed": False, "notes": ["Reasoning engine unavailable."]}

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

    def _fetch_news(self, symbol) -> dict:
        """
        Fetches and classifies recent news through the ML pipeline.
        Returns structured article data with sentiment, confidence, and relevance.
        """
        news_strategy = self.sm.engines.get("News_Trading")
        if not news_strategy:
            return {"error": "News strategy not loaded.", "articles": []}
        
        try:
            # 1. Pull recent raw RSS feeds
            articles = news_strategy.fetcher.fetch(max_age_hours=12)
            if not articles:
                return {"articles": []}
            
            # 2. Run them through the ML Classifier pipeline
            classified = []
            for article in articles:
                c = news_strategy.classifier.classify(article, symbol=symbol)
                # Filter out statistical anomalies, noise, and irrelevant pairs
                if not c.is_fake and c.relevance > 0.15:
                    classified.append(c)
            
            # 3. Rank by highest confidence
            classified.sort(key=lambda x: x.confidence, reverse=True)
            
            # 4. Convert to dict for JSON serialization
            from dataclasses import asdict
            result_articles = []
            for c in classified[:5]:  # Top 5 articles
                article_dict = asdict(c)
                result_articles.append(article_dict)
            
            return {"articles": result_articles}
        except Exception as e:
            return {"error": str(e), "articles": []}

    def _risk_check(self, symbol) -> dict:
        r   = profile.risk(symbol)
        cfg = {
            "max_daily_loss": r.max_daily_loss, 
            "risk_percentage": r.risk_pct,
            "stop_loss_pips": r.stop_loss_pips,
            "trading_symbols": profile.symbols(),
            }

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
        r = profile.risk(symbol)
        return self.rm.calculate_safe_trade(
            symbol         = symbol,
            base_risk_pct  = r.risk_pct,
            stop_loss_pips = r.stop_loss_pips,
            max_daily_loss = r.max_daily_loss,
            portfolio_size = len(profile.symbols()),
        )

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
        tracked = profile.symbols()
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

        tracked = profile.symbols()
        cooldowns = {}
        for symbol in tracked:
            in_cd, remaining = self.broker.is_in_cooldown(symbol)
            if in_cd:
                cooldowns[symbol] = f"{remaining:.0f}s"

        return {
            "balance":      round(account.balance, 2),
            "equity":       round(account.equity, 2),
            "profit":       round(account.profit, 2),
            "margin_level": round(account.margin_level, 2) if account.margin_level else None,
            "margin":       round(getattr(account, "margin", 0.0), 2),
            "free_margin":  round(getattr(account, "margin_free", 0.0), 2),
            "cooldowns":    cooldowns,
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
        
        learner = getattr(self.sm, "learner", None)
        regime  = learner.get_current_regime() if learner else "Unknown"
        best    = learner.get_best_strategies_for_regime(regime) if learner else []
        return {
            "regime":           regime,
            "best_strategies":  best,
            "tracked_symbols":  profile.symbols(),
            
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
        r = profile.risk()
        
        try:
            results = self.pm.evaluate_portfolio_opportunities(
                risk_pct       = r.risk_pct,
                stop_loss      = r.stop_loss_pips,
                max_daily_loss = r.max_daily_loss,
                dry_run        = True,
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
        r = profile.risk()
        max_loss = r.max_daily_loss
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
                            profit_value = row.get("Profit", "") or row.get("Comment", "")
                            try:
                                if row.get("Profit"):
                                    pnl += float(row.get("Profit"))
                                else:
                                    pnl += float(profit_value.replace("Profit:", "").strip())
                            except ValueError:
                                pass
            return {
                "trades": trades, "pnl": round(pnl, 2),
                "count": len(trades), "buys": buys, "sells": sells, "closes": closes,
            }
        except Exception:
            return {"trades": [], "pnl": 0.0, "count": 0}

    def _performance_analysis(self, plan: AgentPlan) -> dict:
        try:
            import pandas as pd
            if not self.TRADE_HISTORY.exists():
                return {}
            
            df = pd.read_csv(self.TRADE_HISTORY)
            df = df[df['Action'] == 'CLOSE'].copy()
            if df.empty:
                return {}

            # Clean profit values
            def extract_profit(val):
                try: return float(str(val).replace('Profit:', '').strip())
                except: return 0.0
            
            df['CleanProfit'] = df['Profit'].fillna(df['Comment']).apply(extract_profit)
            
            total_trades = len(df)
            wins = len(df[df['CleanProfit'] > 0])
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            

            # Group by Strategy, filter out 'Unknown'
            strat_df = df[df['Strategy'].notna() & (df['Strategy'] != 'Unknown')]
            if not strat_df.empty:
                strat_group = strat_df.groupby('Strategy')['CleanProfit']
                strat_stats = strat_group.agg(['count', lambda x: (x > 0).mean() * 100]).rename(columns={'<lambda_0>': 'win_rate'})
                best_strat  = strat_stats['win_rate'].idxmax()
                worst_strat = strat_stats['win_rate'].idxmin()
            else:
                best_strat  = "N/A — execute via scanner to log strategy"
                worst_strat = "N/A"

            # Group by Symbol
            sym_group = df.groupby('Symbol')['CleanProfit'].sum()
            best_sym = sym_group.idxmax() if not sym_group.empty else "N/A"
            worst_sym = sym_group.idxmin() if not sym_group.empty else "N/A"

            # Profit Factor
            gross_profit = df[df['CleanProfit'] > 0]['CleanProfit'].sum()
            gross_loss = abs(df[df['CleanProfit'] < 0]['CleanProfit'].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_win = df[df['CleanProfit'] > 0]['CleanProfit'].mean()
            avg_loss = df[df['CleanProfit'] < 0]['CleanProfit'].mean()

            return {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 1),
                "best_strategy": best_strat,
                "worst_strategy": worst_strat,
                "best_symbol": f"{best_sym} (+${sym_group.max():.2f})",
                "worst_symbol": f"{worst_sym} (${sym_group.min():.2f})",
                "profit_factor": round(pf, 2),
                "avg_win": round(avg_win, 2) if not pd.isna(avg_win) else 0,
                "avg_loss": round(avg_loss, 2) if not pd.isna(avg_loss) else 0
            }
        except Exception as e:
            return {"error": str(e)}

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
 
        # Suggested action for confirm flow (Win Rate Fix #3)
        # Now allow Grade C at 50% position size (was A/B only)
        # Suggested action for confirm flow (Win Rate Fix #3)
        # GATED: Only propose an actionable execution payload if the original intent requests trading.
        trade_intents = ("execute_trade", "open_buy", "open_sell", "trade_execution")
        
        if (plan.intent in trade_intents and 
            action in ("BUY", "SELL") and 
            grade in ("A", "B", "C") and 
            approved and lots):
            
            trade_lots = lots if grade in ("A", "B") else lots * 0.5
            agreements = sum(
                1 for s in r.get("signal_ensemble", {}).get("strategy_signals", {}).values()
                if s.get("action") == action and s.get("confidence", 0) > 0.4
            )
            if agreements >= 2:  
                plan.suggested_action = f"{action} {trade_lots:.2f} {symbol}"
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
        cds = acct.get("cooldowns", {})
        if cds:
            cd_str = ", ".join(f"{s}({t})" for s, t in cds.items())
            lines.append(f"⏸️ Cooling down: {cd_str}")
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
        perf = r.get("performance_analysis", {})
        
        if not perf or "error" in perf:
            return "Not enough closed trade history to generate a performance report yet."

        best_strat  = perf.get('best_strategy',  'N/A')
        worst_strat = perf.get('worst_strategy', 'N/A')
        if best_strat  == "Unknown": best_strat  = "N/A — label your strategies"
        if worst_strat == "Unknown": worst_strat = "N/A — label your strategies"

        lines = [
            "📊 ARIA Personal Performance Analytics",
            "──────────────────────────────────────",
            f"Win Rate:       {perf.get('win_rate')}% ({perf.get('total_trades')} trades)",
            f"Profit Factor:  {perf.get('profit_factor')}",
            f"Avg Win/Loss:   +${perf.get('avg_win')} / -${abs(perf.get('avg_loss', pd.isna))}",
            "",
            f"Best Strategy:  {best_strat} (Highest Win Rate)",
            f"Worst Strategy: {worst_strat} (Needs Retraining)",
            f"Best Symbol:    {perf.get('best_symbol')}",
            f"Worst Symbol:   {perf.get('worst_symbol')}",
            "──────────────────────────────────────",
            "Use this data to disable underperforming strategies or pairs."
        ]
        return "\n".join(lines)
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
        news_r = r.get("fetch_news", {})
        articles = news_r.get("articles", [])
        error  = news_r.get("error")
        
        if error or not articles:
            return (
                f"Current regime: {regime}. "
                f"No highly relevant or market-moving news detected recently. "
                f"Market seems quiet or news is noise. "
                f"Equity: ${eq:,.2f}."
            )
        
        lines = [f"Market Intelligence (Regime: {regime}):"]
        for article in articles:
            sentiment = article.get("sentiment", "NEUTRAL")
            confidence = article.get("confidence", 0.0)
            cluster = article.get("cluster_label", "Event")
            title = article.get("title", "")
            
            icon = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "⚪"
            
            lines.append(f"• {icon} [{cluster}] {title} (Conf: {confidence:.0%})")
        
        lines.append(f"\nEquity: ${eq:,.2f}.")
        return "\n".join(lines)
 
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

        for step in plan.steps:
            if step.name == "dynamic_risk_targets" and step.result:
                plan.context["dynamic_targets"] = step.result
                break
        
        response = self.synthesizer.synthesize(plan)
        plan.final_response = response
        self.last_plan = plan

        return response

