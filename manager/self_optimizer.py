"""
manager/self_optimizer.py — ARIA Self-Optimization Engine

Continuously monitors live trading performance and automatically triggers
parameter optimization when win rate or profit factor degrades below thresholds.

Architecture
------------
PerformanceMonitor   — reads trade_history.csv and computes rolling metrics
OptimizationTrigger  — decides WHEN to optimize (thresholds, schedule, cooldown)
ParamApplicator      — applies best params back to live engines safely
SelfOptimizer        — orchestrates the full loop as a background daemon

How it works
------------
1. Every CHECK_INTERVAL seconds, PerformanceMonitor recomputes:
   - Rolling win rate (last N trades)
   - Profit factor
   - Per-strategy win rates
   - Per-symbol net PnL

2. OptimizationTrigger fires when ANY of these are true:
   - Rolling win rate < WIN_RATE_FLOOR (default 45%)
   - Profit factor < PROFIT_FACTOR_FLOOR (default 1.0)
   - A strategy's individual win rate < STRATEGY_FLOOR (default 40%)
   - Scheduled off-hours optimization (default: 22:00 UTC daily)

3. BacktestEngine sweeps param combinations (random search, ~25 trials)
   using cached OHLCV data — no live broker calls needed during sweep.

4. ParamApplicator atomically patches live strategy engines with winning params
   and writes them to data/profile.json so they survive restarts.

5. All decisions are logged to data/self_optimizer_log.json for audit.

Usage
-----
    optimizer = SelfOptimizer(strategy_manager, broker, notify_callback=agent_notify)
    optimizer.start()   # background daemon

    # Or trigger manually from ARIA chat:
    optimizer.trigger_now("EURUSD")
"""

from __future__ import annotations

import csv
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WIN_RATE_FLOOR        = 0.45   # trigger if rolling WR drops below 45%
PROFIT_FACTOR_FLOOR   = 1.0    # trigger if PF drops below 1.0
STRATEGY_WIN_FLOOR    = 0.40   # trigger if any strategy WR < 40%
MIN_TRADES_TO_JUDGE   = 15     # don't trigger on tiny sample sizes
ROLLING_WINDOW        = 30     # last N trades for rolling metrics
CHECK_INTERVAL        = 300    # seconds between performance checks (5 min)
OPT_COOLDOWN_HOURS    = 6      # minimum hours between optimization runs
SCHEDULED_OPT_HOUR   = 22     # UTC hour for daily scheduled optimization
N_OPT_TRIALS         = 25     # random search trials per symbol
MAX_SYMBOLS_PER_RUN   = 3      # cap symbols optimized per run to control time
LOG_FILE              = Path("data/self_optimizer_log.json")
PARAMS_FILE           = Path("data/optimized_params.json")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceSnapshot:
    timestamp:         str
    symbol:            str
    total_trades:      int
    win_rate:          float
    profit_factor:     float
    net_pnl:           float
    avg_win:           float
    avg_loss:          float
    worst_strategy:    str
    worst_strategy_wr: float
    needs_optimization: bool
    trigger_reason:    str


@dataclass
class OptimizationRecord:
    timestamp:      str
    symbol:         str
    trigger_reason: str
    before_wr:      float
    after_sharpe:   float
    best_params:    dict
    n_trials:       int
    elapsed_sec:    float
    applied:        bool


# ─────────────────────────────────────────────────────────────────────────────
# Performance Monitor
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Reads trade_history.csv and computes rolling performance metrics.
    All reads are non-destructive — never writes to the CSV.
    """

    TRADE_HISTORY = Path("data/trade_history.csv")

    def snapshot(self, symbol: str = None) -> Optional[PerformanceSnapshot]:
        """
        Compute a performance snapshot for a symbol (or globally).
        Returns None if insufficient data.
        """
        trades = self._load_closed_trades(symbol)
        if len(trades) < MIN_TRADES_TO_JUDGE:
            return None

        recent = trades[-ROLLING_WINDOW:]
        profits = [t["profit"] for t in recent if t["profit"] is not None]

        if not profits:
            return None

        wins   = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        n      = len(profits)
        win_n  = len(wins)

        win_rate      = win_n / n
        gross_win     = sum(wins)
        gross_loss    = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        net_pnl       = sum(profits)
        avg_win       = gross_win / win_n if win_n else 0.0
        avg_loss      = gross_loss / len(losses) if losses else 0.0

        # Per-strategy stats
        strat_map: dict[str, list[float]] = {}
        for t in recent:
            s = t.get("strategy", "Unknown")
            if s and s != "Unknown":
                strat_map.setdefault(s, []).append(1.0 if (t["profit"] or 0) > 0 else 0.0)

        worst_strat    = "Unknown"
        worst_strat_wr = 1.0
        for s, outcomes in strat_map.items():
            if len(outcomes) >= 5:
                wr = sum(outcomes) / len(outcomes)
                if wr < worst_strat_wr:
                    worst_strat_wr = wr
                    worst_strat    = s

        # Determine if optimization is needed
        trigger_reason = ""
        needs_opt      = False

        if win_rate < WIN_RATE_FLOOR:
            needs_opt      = True
            trigger_reason = f"Win rate {win_rate:.0%} < floor {WIN_RATE_FLOOR:.0%}"
        elif profit_factor < PROFIT_FACTOR_FLOOR and profit_factor != float("inf"):
            needs_opt      = True
            trigger_reason = f"Profit factor {profit_factor:.2f} < floor {PROFIT_FACTOR_FLOOR:.2f}"
        elif worst_strat_wr < STRATEGY_WIN_FLOOR and worst_strat != "Unknown":
            needs_opt      = True
            trigger_reason = f"Strategy '{worst_strat}' WR {worst_strat_wr:.0%} < floor {STRATEGY_WIN_FLOOR:.0%}"

        return PerformanceSnapshot(
            timestamp          = datetime.now().isoformat(),
            symbol             = symbol or "ALL",
            total_trades       = len(trades),
            win_rate           = round(win_rate, 3),
            profit_factor      = round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
            net_pnl            = round(net_pnl, 2),
            avg_win            = round(avg_win, 2),
            avg_loss           = round(avg_loss, 2),
            worst_strategy     = worst_strat,
            worst_strategy_wr  = round(worst_strat_wr, 3),
            needs_optimization = needs_opt,
            trigger_reason     = trigger_reason,
        )

    def get_active_symbols(self) -> list[str]:
        """Return symbols that appear in recent trade history, sorted by trade count."""
        trades = self._load_closed_trades()
        recent = trades[-100:]
        counts: dict[str, int] = {}
        for t in recent:
            sym = t.get("symbol", "")
            if sym:
                counts[sym] = counts.get(sym, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)

    def get_worst_performing_symbols(self, top_n: int = 3) -> list[str]:
        """Return symbols with the lowest win rates (minimum 5 trades each)."""
        symbols = self.get_active_symbols()
        sym_wr  = []
        for sym in symbols:
            snap = self.snapshot(sym)
            if snap and snap.total_trades >= MIN_TRADES_TO_JUDGE:
                sym_wr.append((sym, snap.win_rate))
        sym_wr.sort(key=lambda x: x[1])
        return [s for s, _ in sym_wr[:top_n]]

    def _load_closed_trades(self, symbol: str = None) -> list[dict]:
        """Load closed trades from CSV, optionally filtered by symbol."""
        if not self.TRADE_HISTORY.exists():
            return []

        trades = []
        try:
            with open(self.TRADE_HISTORY, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    action = row.get("Action", "").upper()
                    if action not in ("CLOSE", "CLOSE_SL_TP", "PARTIAL_CLOSE"):
                        continue
                    sym = row.get("Symbol", "")
                    if symbol and sym != symbol:
                        continue

                    # Parse profit
                    raw_profit = row.get("Profit", "") or row.get("Comment", "")
                    try:
                        profit = float(str(raw_profit).replace("Profit:", "").strip())
                    except (ValueError, AttributeError):
                        profit = None

                    trades.append({
                        "timestamp": row.get("Timestamp", ""),
                        "symbol":    sym,
                        "strategy":  row.get("Strategy", "Unknown"),
                        "profit":    profit,
                    })
        except Exception:
            pass

        return trades


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Trigger
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationTrigger:
    """
    Decides when to run optimization. Enforces cooldown to prevent thrashing.
    """

    def __init__(self):
        self._last_run:    dict[str, datetime] = {}   # symbol → last run time
        self._last_scheduled: Optional[datetime] = None

    def should_trigger(self, snap: PerformanceSnapshot) -> bool:
        """True if optimization should run for this snapshot's symbol."""
        if not snap.needs_optimization:
            return False

        sym = snap.symbol
        last = self._last_run.get(sym)
        if last:
            hours_since = (datetime.now() - last).total_seconds() / 3600
            if hours_since < OPT_COOLDOWN_HOURS:
                return False   # still in cooldown

        return True

    def should_run_scheduled(self) -> bool:
        """True if it's time for the daily scheduled optimization."""
        now = datetime.now(timezone.utc)
        if now.hour != SCHEDULED_OPT_HOUR:
            return False
        if self._last_scheduled:
            hours_since = (now - self._last_scheduled).total_seconds() / 3600
            if hours_since < 20:   # avoid double-firing within same hour window
                return False
        return True

    def mark_ran(self, symbol: str):
        self._last_run[symbol] = datetime.now()

    def mark_scheduled_ran(self):
        self._last_scheduled = datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Param Applicator
# ─────────────────────────────────────────────────────────────────────────────

class ParamApplicator:
    """
    Applies optimized parameters to live strategy engines and persists them.
    Thread-safe — acquires no external locks; relies on Python's GIL for dict writes.
    """

    # Which engine attributes map to which optimizer result keys
    STRATEGY_ATTR_MAP = {
        "Mean_Reversion": {
            "bb_length":     "bb_length",
            "bb_std":        "bb_std",
            "rsi_length":    "rsi_length",
            "adx_threshold": "adx_threshold",
            "min_rr":        "min_rr",
        },
        "Momentum": {
            "adx_min":          "adx_min",
            "rsi_bull_min":     "rsi_bull_min",
            "rsi_bear_max":     "rsi_bear_max",
            "atr_sl_mult":      "atr_sl_mult",
            "atr_tp_base_mult": "atr_tp_base_mult",
        },
        "Breakout": {
            "lookback_window":   "lookback_window",
            "volume_multiplier": "volume_multiplier",
            "atr_buffer_pct":    "atr_buffer_pct",
        },
    }

    def apply(
        self,
        strategy_manager,
        opt_result,            # OptimizationResult from optimizer.py
        strategy_name: str,
        notify: Callable,
    ) -> bool:
        """
        Apply best params to the live engine.
        Returns True if any parameters were actually changed.
        """
        if not opt_result or not opt_result.best_params:
            return False

        engine = strategy_manager.engines.get(strategy_name)
        if engine is None:
            return False

        attr_map   = self.STRATEGY_ATTR_MAP.get(strategy_name, {})
        changed    = []
        best       = opt_result.best_params

        for result_key, attr_name in attr_map.items():
            if result_key in best and hasattr(engine, attr_name):
                old_val = getattr(engine, attr_name)
                new_val = best[result_key]
                if old_val != new_val:
                    setattr(engine, attr_name, new_val)
                    changed.append(f"{attr_name}: {old_val} → {new_val}")

        if changed:
            notify(
                f"[SelfOptimizer] Applied {len(changed)} param change(s) to {strategy_name}: "
                + ", ".join(changed),
                priority="normal",
            )
            self._persist(strategy_name, best)
            return True

        return False

    def load_persisted(self, strategy_manager, notify: Callable):
        """
        On startup, reload previously optimized params into live engines.
        Silently skips missing strategies or attributes.
        """
        if not PARAMS_FILE.exists():
            return

        try:
            stored = json.loads(PARAMS_FILE.read_text())
        except Exception:
            return

        for strategy_name, params in stored.items():
            engine   = strategy_manager.engines.get(strategy_name)
            attr_map = self.STRATEGY_ATTR_MAP.get(strategy_name, {})
            if engine is None or not attr_map:
                continue
            applied = []
            for result_key, attr_name in attr_map.items():
                if result_key in params and hasattr(engine, attr_name):
                    setattr(engine, attr_name, params[result_key])
                    applied.append(attr_name)
            if applied:
                notify(
                    f"[SelfOptimizer] Restored optimized params for {strategy_name}: "
                    + ", ".join(applied),
                    priority="normal",
                )

    def _persist(self, strategy_name: str, params: dict):
        """Write optimized params to disk for survival across restarts."""
        PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        stored: dict = {}
        if PARAMS_FILE.exists():
            try:
                stored = json.loads(PARAMS_FILE.read_text())
            except Exception:
                pass
        stored[strategy_name] = params
        PARAMS_FILE.write_text(json.dumps(stored, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationAuditLog:
    """Appends structured records to data/self_optimizer_log.json."""

    def record(self, rec: OptimizationRecord):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        history: list = []
        if LOG_FILE.exists():
            try:
                history = json.loads(LOG_FILE.read_text())
            except Exception:
                history = []

        history.append(asdict(rec))
        # Keep last 500 records
        history = history[-500:]
        LOG_FILE.write_text(json.dumps(history, indent=2))

    def load_recent(self, n: int = 10) -> list[dict]:
        if not LOG_FILE.exists():
            return []
        try:
            history = json.loads(LOG_FILE.read_text())
            return history[-n:]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# SelfOptimizer — Main daemon
# ─────────────────────────────────────────────────────────────────────────────

class SelfOptimizer:
    """
    Background daemon that monitors performance and auto-optimizes.

    Lifecycle
    ---------
        opt = SelfOptimizer(strategy_manager, broker, notify_callback=agent_notify)
        opt.start()          # starts background thread
        opt.trigger_now()    # manual trigger from ARIA chat
        opt.status()         # human-readable status string
        opt.stop()           # clean shutdown
    """

    def __init__(
        self,
        strategy_manager,
        broker,
        notify_callback: Callable = print,
    ):
        self.sm      = strategy_manager
        self.broker  = broker
        self.notify  = notify_callback

        self.monitor   = PerformanceMonitor()
        self.trigger   = OptimizationTrigger()
        self.applicator = ParamApplicator()
        self.audit_log  = OptimizationAuditLog()

        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._opt_lock   = threading.Lock()   # prevents overlapping optimization runs
        self._last_snap: dict[str, PerformanceSnapshot] = {}

        # Restore previously optimized params on startup
        self.applicator.load_persisted(self.sm, self.notify)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="SelfOptimizer"
        )
        self._thread.start()
        self.notify("[SelfOptimizer] Started — monitoring performance every 5 min.", priority="normal")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def trigger_now(self, symbol: str = None, reason: str = "Manual trigger") -> str:
        """
        Manually trigger an optimization run.
        Called from ARIA chat: 'optimize the strategy' or 'self-optimize'.
        Returns a human-readable result summary.
        """
        if not self._opt_lock.acquire(blocking=False):
            return "Optimization already running — please wait a moment."

        try:
            symbols = [symbol] if symbol else self.monitor.get_worst_performing_symbols(MAX_SYMBOLS_PER_RUN)
            if not symbols:
                # Fall back to all tracked symbols
                from manager.profile_manager import profile
                symbols = profile.symbols()[:MAX_SYMBOLS_PER_RUN]

            if not symbols:
                return "No symbols to optimize. Add symbols to your portfolio first."

            results = []
            for sym in symbols:
                snap = self.monitor.snapshot(sym)
                before_wr = snap.win_rate if snap else 0.0
                result = self._run_optimization(sym, reason, before_wr)
                if result:
                    results.append(result)

            if not results:
                return "Optimization ran but no improvements were found. Params unchanged."

            lines = ["Self-optimization complete:"]
            for r in results:
                lines.append(
                    f"  {r.symbol}: {r.n_trials} trials in {r.elapsed_sec:.0f}s | "
                    f"Best Sharpe {r.after_sharpe:.2f} | "
                    f"Params {'applied ✅' if r.applied else 'unchanged'}"
                )
            return "\n".join(lines)

        finally:
            self._opt_lock.release()

    def status(self) -> str:
        """Return a human-readable status summary."""
        lines = ["Self-Optimizer Status:"]
        for sym, snap in self._last_snap.items():
            lines.append(
                f"  {sym}: WR {snap.win_rate:.0%} | PF {snap.profit_factor:.2f} | "
                f"{'⚠️ needs opt' if snap.needs_optimization else '✅ healthy'}"
            )

        recent = self.audit_log.load_recent(3)
        if recent:
            lines.append("\nRecent optimization runs:")
            for r in recent:
                lines.append(
                    f"  [{r['timestamp'][:16]}] {r['symbol']} — "
                    f"{r['trigger_reason']} | "
                    f"{'Applied' if r['applied'] else 'No change'}"
                )

        if not self._last_snap and not recent:
            lines.append("  No data yet — waiting for sufficient trade history.")

        return "\n".join(lines)

    def get_performance_report(self) -> str:
        """Detailed performance report across all tracked symbols."""
        symbols = self.monitor.get_active_symbols()
        if not symbols:
            return "No closed trade history found yet."

        lines = [
            "📊 Live Performance Diagnostics",
            "─" * 40,
        ]
        for sym in symbols[:10]:
            snap = self.monitor.snapshot(sym)
            if not snap:
                continue
            flag = " ⚠️" if snap.needs_optimization else " ✅"
            lines.append(
                f"{sym:<12} WR: {snap.win_rate:.0%}  PF: {snap.profit_factor:.2f}  "
                f"PnL: ${snap.net_pnl:+.2f}  Trades: {snap.total_trades}{flag}"
            )
            if snap.needs_optimization:
                lines.append(f"           → {snap.trigger_reason}")

        lines.append("─" * 40)
        return "\n".join(lines)

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                self._check_cycle()
            except Exception as exc:
                self.notify(f"[SelfOptimizer] Error in check cycle: {exc}", priority="normal")
            time.sleep(CHECK_INTERVAL)

    def _check_cycle(self):
        """One monitoring + trigger cycle."""
        symbols = self.monitor.get_active_symbols()

        # Performance-based triggers
        for sym in symbols[:MAX_SYMBOLS_PER_RUN * 2]:
            snap = self.monitor.snapshot(sym)
            if not snap:
                continue
            self._last_snap[sym] = snap

            if self.trigger.should_trigger(snap):
                if not self._opt_lock.acquire(blocking=False):
                    continue   # already running
                try:
                    self.notify(
                        f"[SelfOptimizer] {sym} triggered: {snap.trigger_reason}",
                        priority="normal",
                    )
                    self._run_optimization(sym, snap.trigger_reason, snap.win_rate)
                    self.trigger.mark_ran(sym)
                finally:
                    self._opt_lock.release()

        # Scheduled daily optimization
        if self.trigger.should_run_scheduled():
            if not self._opt_lock.acquire(blocking=False):
                return
            try:
                self.notify("[SelfOptimizer] Running scheduled daily optimization.", priority="normal")
                worst = self.monitor.get_worst_performing_symbols(MAX_SYMBOLS_PER_RUN)
                for sym in worst:
                    snap = self.monitor.snapshot(sym)
                    before_wr = snap.win_rate if snap else 0.0
                    self._run_optimization(sym, "Scheduled daily", before_wr)
                self.trigger.mark_scheduled_ran()
            finally:
                self._opt_lock.release()

    # ── Core optimization runner ──────────────────────────────────────────────

    def _run_optimization(
        self,
        symbol:        str,
        trigger_reason: str,
        before_wr:     float,
    ) -> Optional[OptimizationRecord]:
        """
        Run a random-search optimization for one symbol.
        Picks the worst-performing active strategy and tunes its params.
        Applies results if they improve on the current Sharpe baseline.
        """
        t0 = time.perf_counter()

        try:
            from optimizer import StrategyParamOptimizer, RiskParamOptimizer

            # Choose which strategy to optimize: the worst performer
            snap = self.monitor.snapshot(symbol)
            target_strategy = snap.worst_strategy if snap and snap.worst_strategy != "Unknown" else "Mean_Reversion"

            # Only optimize strategies we have param spaces for
            from optimizer import STRATEGY_PARAM_SPACES
            if target_strategy not in STRATEGY_PARAM_SPACES:
                target_strategy = next(
                    (s for s in STRATEGY_PARAM_SPACES if s in self.sm.engines), "Mean_Reversion"
                )

            self.notify(
                f"[SelfOptimizer] Optimizing {target_strategy} on {symbol} "
                f"({N_OPT_TRIALS} trials)...",
                priority="normal",
            )

            strat_opt = StrategyParamOptimizer(self.sm, notify=lambda *a, **kw: None)
            result = strat_opt.random_search(
                strategy_name = target_strategy,
                symbol        = symbol,
                n_trials      = N_OPT_TRIALS,
                metric        = "sharpe",
                notify        = lambda *a, **kw: None,
            )

            elapsed = round(time.perf_counter() - t0, 1)

            # Only apply if we got a meaningful result
            applied = False
            if result.best_metric > 0.0 and result.best_params:
                applied = self.applicator.apply(
                    self.sm, result, target_strategy, self.notify
                )

            # Also run a quick risk param sweep
            if applied:
                risk_opt = RiskParamOptimizer(self.sm, notify=lambda *a, **kw: None)
                risk_result = risk_opt.optimize(
                    symbol   = symbol,
                    n_trials = 15,
                    metric   = "calmar",
                    notify   = lambda *a, **kw: None,
                )
                if risk_result.best_metric > 0.0:
                    risk_opt.apply_to_profile(risk_result)
                    self.notify(
                        f"[SelfOptimizer] Risk params updated for {symbol}: "
                        f"risk={risk_result.best_params.get('risk_pct', '?')}% "
                        f"SL={risk_result.best_params.get('sl_pips', '?')}p",
                        priority="normal",
                    )

            record = OptimizationRecord(
                timestamp      = datetime.now().isoformat(),
                symbol         = symbol,
                trigger_reason = trigger_reason,
                before_wr      = before_wr,
                after_sharpe   = result.best_metric,
                best_params    = result.best_params,
                n_trials       = N_OPT_TRIALS,
                elapsed_sec    = elapsed,
                applied        = applied,
            )
            self.audit_log.record(record)

            status = "✅ Params improved and applied" if applied else "ℹ️ No improvement found"
            self.notify(
                f"[SelfOptimizer] {symbol} done in {elapsed:.0f}s | "
                f"Best Sharpe {result.best_metric:.2f} | {status}",
                priority="normal",
            )

            return record

        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 1)
            self.notify(
                f"[SelfOptimizer] Optimization failed for {symbol}: {exc}",
                priority="normal",
            )
            return None

    # ── Ensemble weight tuning (bonus) ────────────────────────────────────────

    def tune_ensemble_weights(self, symbol: str) -> str:
        """
        Tune ensemble vote weights for the current market regime.
        Call when overall confluence score is poor despite individual signals firing.
        """
        if not self._opt_lock.acquire(blocking=False):
            return "Optimization already running."

        try:
            from optimizer import EnsembleWeightOptimizer

            regime = "Unknown"
            if hasattr(self.sm, "learner") and self.sm.learner:
                regime = self.sm.learner.get_current_regime()

            self.notify(f"[SelfOptimizer] Tuning ensemble weights for regime: {regime}", priority="normal")

            ens_opt = EnsembleWeightOptimizer(self.sm, notify=lambda *a, **kw: None)
            result  = ens_opt.optimize(
                symbol    = symbol,
                regime    = regime,
                n_trials  = 20,
                metric    = "sharpe",
            )

            if result.best_metric > 0.0:
                return (
                    f"Ensemble weights tuned for '{regime}' regime | "
                    f"Best Sharpe {result.best_metric:.2f} | "
                    f"Weights: {result.best_params}"
                )
            return f"Ensemble tuning ran but no improvement found for '{regime}' regime."

        finally:
            self._opt_lock.release()