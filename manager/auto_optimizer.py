"""
manager/auto_optimizer.py — ARIA Autonomous Optimization Engine

Runs continuously in the background. No triggers, no thresholds to wait for.
Learns from every closed trade and adapts parameters in real-time.

Three interlocking loops
------------------------
Loop 1 — MicroAdaptor (every 60s)
    Reads the last 20 closed trades. If a strategy's win rate over that window
    is trending down, nudges its confidence threshold up by a small delta.
    This is a fast, lightweight signal filter — no backtesting needed.

Loop 2 — StrategyTuner (every 4h or after 10 consecutive losses)
    Runs a 30-trial random-search backtest on the worst-performing strategy
    for the most-traded symbol. Applies best params if Sharpe improves > 0.1.

Loop 3 — EnsembleCalibrator (every 24h, 22:00 UTC)
    Rebalances regime weights in MetaScorer using the last 30 days of
    trade outcomes. Strategies that underperformed their regime get downweighted.
    Uses the UnsupervisedLearner's regime labels to match trades to regimes.

Architecture principles
-----------------------
- Never touches live broker during optimization sweeps (uses LocalCache)
- All writes are atomic (write temp file → rename)
- Each loop runs in its own daemon thread — no blocking
- Graceful fallback: if backtest data is insufficient, MicroAdaptor still runs
- All changes logged to data/auto_optimizer_log.jsonl (append-only)

Usage
-----
    from manager.auto_optimizer import AutoOptimizer

    auto_opt = AutoOptimizer(strategy_manager, notify_callback=agent_notify)
    auto_opt.start()    # starts all three loops

    # From ARIA chat:
    auto_opt.report()   # human-readable status
    auto_opt.force_tune("EURUSD")   # immediate strategy tune
"""

from __future__ import annotations

import csv
import json
import math
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE     = Path("data/auto_optimizer_log.jsonl")
STATE_FILE   = Path("data/auto_optimizer_state.json")
PARAMS_FILE  = Path("data/optimized_params.json")
TRADE_CSV    = Path("data/trade_history.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_closed_trades(symbol: str = None, limit: int = 200) -> list[dict]:
    if not TRADE_CSV.exists():
        return []
    rows = []
    try:
        with open(TRADE_CSV, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                action = row.get("Action", "").upper()
                if action not in ("CLOSE", "CLOSE_SL_TP"):
                    continue
                sym = row.get("Symbol", "")
                if symbol and sym != symbol:
                    continue
                raw = row.get("Profit", "") or row.get("Comment", "")
                try:
                    profit = float(str(raw).replace("Profit:", "").strip())
                except Exception:
                    continue
                rows.append({
                    "timestamp": row.get("Timestamp", ""),
                    "symbol":    sym,
                    "strategy":  row.get("Strategy", "Unknown"),
                    "profit":    profit,
                })
    except Exception:
        pass
    return rows[-limit:]


def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("profit") or 0) > 0)
    return wins / len(trades)


def _profit_factor(trades: list[dict]) -> float:
    wins   = sum(t["profit"] for t in trades if (t.get("profit") or 0) > 0)
    losses = abs(sum(t["profit"] for t in trades if (t.get("profit") or 0) < 0))
    return wins / losses if losses > 0 else float("inf")


def _append_log(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Loop 1 — MicroAdaptor
# ─────────────────────────────────────────────────────────────────────────────

class MicroAdaptor:
    """
    Fast confidence-threshold nudger. Runs every 60 seconds.

    For each strategy, keeps a rolling 20-trade window. If win rate over
    that window is below 0.48, raises the strategy's internal confidence
    threshold by NUDGE_DELTA (making it more selective). If win rate is
    above 0.62, lowers it slightly (making it more aggressive).

    This has NO backtesting cost — just reading CSV and tweaking one float.
    """

    INTERVAL      = 60        # seconds
    WINDOW        = 20        # trades in rolling window
    NUDGE_DELTA   = 0.02      # how much to move the threshold per cycle
    FLOOR         = 0.50      # minimum confidence threshold
    CEILING       = 0.85      # maximum confidence threshold
    WR_TOO_LOW    = 0.45
    WR_TOO_HIGH   = 0.62

    # Strategy attribute that acts as the main quality gate
    THRESHOLD_ATTR = {
        "Mean_Reversion": "min_rr",          # tighten R:R gate
        "Momentum":       "adx_min",         # tighten trend filter
        "Breakout":       "volume_multiplier",
        "Scalping":       None,
        "Trend_Following":"adx_threshold",
    }

    # Nudge direction for each attribute (up = more selective)
    NUDGE_UP_MEANS = {
        "min_rr":           +0.05,   # higher R:R = harder to enter
        "adx_min":          +1.0,    # higher ADX = stricter trend
        "volume_multiplier":+0.05,
        "adx_threshold":    +1.0,
    }

    def __init__(self, strategy_manager, notify: Callable):
        self.sm     = strategy_manager
        self.notify = notify
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MicroAdaptor")
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self._cycle()
            except Exception as exc:
                pass   # never crash the daemon
            time.sleep(self.INTERVAL)

    def _cycle(self):
        trades = _load_closed_trades(limit=200)
        if len(trades) < 10:
            return

        # Group by strategy
        by_strategy: dict[str, list] = defaultdict(list)
        for t in trades:
            s = t.get("strategy", "Unknown")
            if s and s != "Unknown":
                by_strategy[s].append(t)

        for strat_name, strat_trades in by_strategy.items():
            recent = strat_trades[-self.WINDOW:]
            if len(recent) < 8:
                continue

            wr     = _win_rate(recent)
            engine = self.sm.engines.get(strat_name)
            attr   = self.THRESHOLD_ATTR.get(strat_name)

            if engine is None or attr is None:
                continue
            if not hasattr(engine, attr):
                continue

            current = float(getattr(engine, attr))
            nudge   = self.NUDGE_UP_MEANS.get(attr, 0.02)

            if wr < self.WR_TOO_LOW:
                new_val = min(current + nudge, current * 1.15)
                setattr(engine, attr, round(new_val, 4))
                self._log("nudge_up", strat_name, attr, current, new_val, wr)

            elif wr > self.WR_TOO_HIGH:
                new_val = max(current - nudge * 0.5, current * 0.92)
                setattr(engine, attr, round(new_val, 4))
                self._log("nudge_down", strat_name, attr, current, new_val, wr)

    def _log(self, direction, strategy, attr, old, new, wr):
        _append_log({
            "loop": "MicroAdaptor",
            "ts": datetime.now().isoformat(),
            "direction": direction,
            "strategy": strategy,
            "attr": attr,
            "old": round(old, 4),
            "new": round(new, 4),
            "rolling_wr": round(wr, 3),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Loop 2 — StrategyTuner
# ─────────────────────────────────────────────────────────────────────────────

class StrategyTuner:
    """
    Full backtest-based parameter search. Runs every 4 hours, or immediately
    after 10 consecutive losses on any symbol.

    Uses the existing optimizer.py random_search — 30 trials per run.
    Only applies params if the new Sharpe > current_sharpe + 0.1.
    """

    INTERVAL           = 4 * 3600   # 4 hours
    CONSECUTIVE_LOSS_TRIGGER = 8    # trigger immediately after N consecutive losses
    N_TRIALS           = 30
    MIN_SHARPE_GAIN    = 0.10       # only apply if Sharpe improves by this much

    def __init__(self, strategy_manager, notify: Callable):
        self.sm         = strategy_manager
        self.notify     = notify
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._opt_lock  = threading.Lock()
        self._last_run: dict[str, float] = {}    # symbol → epoch time
        self._consec_losses: dict[str, int] = {} # symbol → count

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="StrategyTuner")
        self._thread.start()

    def stop(self):
        self._running = False

    def record_outcome(self, symbol: str, profit: float):
        """Call this after every close to track consecutive losses."""
        if profit < 0:
            self._consec_losses[symbol] = self._consec_losses.get(symbol, 0) + 1
            if self._consec_losses[symbol] >= self.CONSECUTIVE_LOSS_TRIGGER:
                self._consec_losses[symbol] = 0
                threading.Thread(
                    target=self._tune_symbol,
                    args=(symbol, "consecutive_loss_trigger"),
                    daemon=True,
                ).start()
        else:
            self._consec_losses[symbol] = 0

    def force_tune(self, symbol: str) -> str:
        """Manually trigger tuning for a symbol. Returns summary."""
        if not self._opt_lock.acquire(blocking=False):
            return "Tuning already in progress — please wait."
        try:
            result = self._tune_symbol(symbol, "manual")
            return result or "Tuning complete."
        finally:
            self._opt_lock.release()

    def _loop(self):
        time.sleep(300)  # warm-up delay
        while self._running:
            try:
                self._scheduled_cycle()
            except Exception:
                pass
            time.sleep(self.INTERVAL)

    def _scheduled_cycle(self):
        """Pick the worst symbol and tune it."""
        trades = _load_closed_trades(limit=300)
        if len(trades) < 20:
            return

        # Find worst symbol by recent win rate
        by_sym: dict[str, list] = defaultdict(list)
        for t in trades[-100:]:
            by_sym[t["symbol"]].append(t)

        worst_sym  = None
        worst_wr   = 1.0
        for sym, sym_trades in by_sym.items():
            if len(sym_trades) < 10:
                continue
            wr = _win_rate(sym_trades)
            if wr < worst_wr:
                worst_wr  = wr
                worst_sym = sym

        if worst_sym:
            self._tune_symbol(worst_sym, f"scheduled (WR {worst_wr:.0%})")

    def _tune_symbol(self, symbol: str, reason: str) -> Optional[str]:
        if not self._opt_lock.acquire(blocking=False):
            return None
        try:
            now = time.time()
            last = self._last_run.get(symbol, 0)
            if now - last < 3600:   # minimum 1h between runs for same symbol
                return None

            self.notify(
                f"[AutoOptimizer] Tuning {symbol} — {reason} ({self.N_TRIALS} trials)...",
                priority="normal",
            )

            try:
                from optimizer import StrategyParamOptimizer, STRATEGY_PARAM_SPACES

                # Find which strategy has most trades on this symbol
                trades = _load_closed_trades(symbol=symbol, limit=100)
                by_strat: dict[str, int] = defaultdict(int)
                for t in trades:
                    s = t.get("strategy", "")
                    if s and s != "Unknown":
                        by_strat[s] += 1

                target = max(by_strat, key=by_strat.get) if by_strat else "Mean_Reversion"
                if target not in STRATEGY_PARAM_SPACES:
                    target = next(
                        (s for s in STRATEGY_PARAM_SPACES if s in self.sm.engines),
                        "Mean_Reversion",
                    )

                opt = StrategyParamOptimizer(self.sm, notify=lambda *a, **kw: None)
                result = opt.random_search(
                    strategy_name = target,
                    symbol        = symbol,
                    n_trials      = self.N_TRIALS,
                    metric        = "sharpe",
                    notify        = lambda *a, **kw: None,
                )

                applied = False
                if result.best_metric > self.MIN_SHARPE_GAIN and result.best_params:
                    # Apply params
                    engine = self.sm.engines.get(target)
                    if engine:
                        for k, v in result.best_params.items():
                            if hasattr(engine, k):
                                setattr(engine, k, v)
                        applied = True
                        self._persist_params(target, result.best_params)

                self._last_run[symbol] = now
                _append_log({
                    "loop":    "StrategyTuner",
                    "ts":      datetime.now().isoformat(),
                    "symbol":  symbol,
                    "reason":  reason,
                    "target":  target,
                    "sharpe":  round(result.best_metric, 3),
                    "applied": applied,
                    "params":  result.best_params,
                })

                status = "✅ applied" if applied else "ℹ️ no gain"
                msg = (
                    f"[AutoOptimizer] {symbol} / {target}: "
                    f"Sharpe {result.best_metric:.2f} | {status}"
                )
                self.notify(msg, priority="normal")
                return msg

            except Exception as exc:
                _append_log({"loop": "StrategyTuner", "ts": datetime.now().isoformat(),
                              "symbol": symbol, "error": str(exc)})
                return None
        finally:
            self._opt_lock.release()

    def _persist_params(self, strategy: str, params: dict):
        stored = {}
        if PARAMS_FILE.exists():
            try:
                stored = json.loads(PARAMS_FILE.read_text())
            except Exception:
                pass
        stored[strategy] = params
        _atomic_write(PARAMS_FILE, stored)


# ─────────────────────────────────────────────────────────────────────────────
# Loop 3 — EnsembleCalibrator
# ─────────────────────────────────────────────────────────────────────────────

def _load_closed_trades_with_regime(limit: int = 500) -> list[dict]:
    """
    Extended version of _load_closed_trades that also reads the
    optional 'Regime' column written by check_signals().
    Falls back to empty string when the column is absent.
    """
    if not TRADE_CSV.exists():
        return []
    rows = []
    try:
        with open(TRADE_CSV, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                action = row.get("Action", "").upper()
                if action not in ("CLOSE", "CLOSE_SL_TP"):
                    continue
                raw = row.get("Profit", "") or row.get("Comment", "")
                try:
                    profit = float(str(raw).replace("Profit:", "").strip())
                except Exception:
                    continue
                rows.append({
                    "timestamp": row.get("Timestamp", ""),
                    "symbol":    row.get("Symbol", ""),
                    "strategy":  row.get("Strategy", "Unknown"),
                    "profit":    profit,
                    # FIX: read the regime that was active when the trade opened.
                    # Empty string means the column didn't exist (legacy CSV).
                    "regime":    row.get("Regime", ""),
                })
    except Exception:
        pass
    return rows[-limit:]
 
def _append_log(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
 
class EnsembleCalibrator:
    """
    Daily rebalance of MetaScorer's REGIME_WEIGHTS based on actual outcomes.

    For each (regime, strategy) pair observed in trade history, computes
    empirical win rate over the last 30 days. Strategies that beat the
    average get upweighted; those that underperform get downweighted.

    Weight formula:
        new_weight = base_weight × (empirical_wr / regime_average_wr)
        clipped to [0.2, 4.0]
    """

    SCHEDULE_HOUR   = 22       # UTC hour for daily run
    LOOKBACK_DAYS   = 30
    MIN_WEIGHT      = 0.2
    MAX_WEIGHT      = 4.0
    MIN_TRADES_REQ  = 5        # minimum trades per (regime, strategy) to adjust

    def __init__(self, strategy_manager, notify: Callable):
        self.sm     = strategy_manager
        self.notify = notify
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: Optional[datetime] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EnsembleCalibrator")
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == self.SCHEDULE_HOUR:
                    if not self._last_run or (now_utc - self._last_run).total_seconds() > 20 * 3600:
                        self._calibrate()
                        self._last_run = now_utc
            except Exception:
                pass
            time.sleep(600)   # check every 10 min

    def force_calibrate(self) -> str:
        """Manually trigger calibration."""
        return self._calibrate()

    def _calibrate(self, learner, strategy_manager) -> str:
        trades = _load_closed_trades_with_regime(limit=500)
        if len(trades) < 30:
            return "Insufficient trade history for ensemble calibration."
 
        current_regime_fallback = (
            learner.get_current_regime() if learner else "Unknown"
        )
 
        # Build (regime, strategy) → [profit] map using per-trade regime
        regime_strat: dict[tuple, list] = defaultdict(list)
        for t in trades:
            strat  = t.get("strategy", "Unknown")
            profit = t.get("profit", 0) or 0
            # FIX: use the regime stored with this trade, not today's regime
            regime = t.get("regime") or current_regime_fallback
            if regime:
                regime_strat[(regime, strat)].append(profit)
 
        try:
            from strategies.models.meta_scorer import REGIME_WEIGHTS
            import strategies.models.meta_scorer as ms_module
        except ImportError:
            return "Could not load MetaScorer module."
 
        changes = []
        for regime, strat_weights in REGIME_WEIGHTS.items():
            regime_trades = [
                v for (r, s), vs in regime_strat.items()
                if r == regime for v in vs
            ]
            if not regime_trades:
                continue
 
            regime_wr = sum(1 for p in regime_trades if p > 0) / len(regime_trades)
            if regime_wr == 0:
                continue
 
            for strat, current_w in strat_weights.items():
                strat_trades = regime_strat.get((regime, strat), [])
                if len(strat_trades) < self.MIN_TRADES_REQ:
                    continue
 
                strat_wr = sum(1 for p in strat_trades if p > 0) / len(strat_trades)
                ratio    = strat_wr / regime_wr
                new_w    = round(
                    max(self.MIN_WEIGHT, min(current_w * ratio, self.MAX_WEIGHT)), 2
                )
 
                if abs(new_w - current_w) > 0.05:
                    ms_module.REGIME_WEIGHTS[regime][strat] = new_w
                    changes.append(f"{regime}/{strat}: {current_w:.1f}→{new_w:.1f}")
 
        _append_log({
            "loop":    "EnsembleCalibrator",
            "ts":      datetime.now().isoformat(),
            "changes": changes,
        })
 
        if changes:
            return (
                f"Ensemble calibrated. {len(changes)} weight change(s): "
                + ", ".join(changes[:5])
            )
        return "Ensemble calibration ran — no significant weight changes needed."
    
# ─────────────────────────────────────────────────────────────────────────────
# AutoOptimizer — orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AutoOptimizer:
    """
    Starts and coordinates all three optimization loops.

    Drop-in replacement / companion to SelfOptimizer.
    Can run alongside it, or instead of it.

    Usage
    -----
        auto_opt = AutoOptimizer(strategy_manager, notify_callback=agent_notify)
        auto_opt.start()

        # integrate with trade close events:
        auto_opt.on_trade_closed(symbol="EURUSD", profit=-12.5)

        # from ARIA chat:
        auto_opt.report()
        auto_opt.force_tune("EURUSD")
        auto_opt.force_calibrate()
    """

    def __init__(self, strategy_manager, notify_callback: Callable = print):
        self.sm     = strategy_manager
        self.notify = notify_callback

        self.micro      = MicroAdaptor(strategy_manager, notify_callback)
        self.tuner      = StrategyTuner(strategy_manager, notify_callback)
        self.calibrator = EnsembleCalibrator(strategy_manager, notify_callback)

        self._load_persisted_params()

    def start(self):
        """Start all three optimization loops."""
        self.micro.start()
        self.tuner.start()
        self.calibrator.start()
        self.notify(
            "[AutoOptimizer] All three loops started: MicroAdaptor (60s), "
            "StrategyTuner (4h), EnsembleCalibrator (daily 22:00 UTC).",
            priority="normal",
        )

    def stop(self):
        self.micro.stop()
        self.tuner.stop()
        self.calibrator.stop()

    def on_trade_closed(self, symbol: str, profit: float):
        """
        Hook this into your trade-close callback so the tuner can react
        to consecutive losses in real time.

        In chat.py ARIA._on_external_close():
            self.auto_optimizer.on_trade_closed(symbol, profit)
        """
        self.tuner.record_outcome(symbol, profit)

    def force_tune(self, symbol: str = None) -> str:
        """Immediate strategy tune for a symbol (or worst-performing if None)."""
        if symbol is None:
            trades = _load_closed_trades(limit=200)
            by_sym: dict[str, list] = defaultdict(list)
            for t in trades[-100:]:
                by_sym[t["symbol"]].append(t)
            symbol = min(by_sym, key=lambda s: _win_rate(by_sym[s])) if by_sym else "EURUSD"
        return self.tuner.force_tune(symbol)

    def force_calibrate(self) -> str:
        """Immediate ensemble weight recalibration."""
        return self.calibrator.force_calibrate()

    def report(self) -> str:
        """Human-readable status report."""
        trades = _load_closed_trades(limit=300)
        if not trades:
            return "No closed trade history yet — auto-optimizer is running and will adapt once trades close."

        recent = trades[-50:]
        overall_wr  = _win_rate(recent)
        overall_pf  = _profit_factor(recent)

        by_strat: dict[str, list] = defaultdict(list)
        by_sym:   dict[str, list] = defaultdict(list)
        for t in recent:
            s = t.get("strategy", "Unknown")
            if s and s != "Unknown":
                by_strat[s].append(t)
            by_sym[t["symbol"]].append(t)

        lines = [
            "🤖 AutoOptimizer — Live Status",
            "─" * 38,
            f"Overall (last {len(recent)} trades):  WR {overall_wr:.0%}  |  PF {overall_pf:.2f}",
            "",
            "By strategy:",
        ]
        for s, ts in sorted(by_strat.items(), key=lambda x: _win_rate(x[1])):
            wr  = _win_rate(ts)
            pf  = _profit_factor(ts)
            eng = self.sm.engines.get(s)
            # Show the key quality gate value if available
            gate = ""
            for attr in ("adx_min", "adx_threshold", "min_rr", "volume_multiplier"):
                if eng and hasattr(eng, attr):
                    gate = f"  [{attr}={getattr(eng, attr):.2f}]"
                    break
            flag = " ⚠️" if wr < 0.45 else " ✅"
            lines.append(f"  {s:<22} WR {wr:.0%}  PF {pf:.2f}{gate}{flag}")

        lines.append("")
        lines.append("By symbol (worst first):")
        sym_sorted = sorted(by_sym.items(), key=lambda x: _win_rate(x[1]))
        for sym, ts in sym_sorted[:5]:
            wr = _win_rate(ts)
            flag = " ⚠️" if wr < 0.45 else " ✅"
            lines.append(f"  {sym:<14} WR {wr:.0%}  ({len(ts)} trades){flag}")

        # Recent log entries
        if LOG_FILE.exists():
            try:
                log_lines = LOG_FILE.read_text().strip().splitlines()[-5:]
                if log_lines:
                    lines += ["", "Recent actions:"]
                    for ll in log_lines:
                        try:
                            rec = json.loads(ll)
                            loop = rec.get("loop", "?")
                            ts   = rec.get("ts", "")[:16]
                            detail = ""
                            if loop == "MicroAdaptor":
                                detail = f"{rec.get('direction','')} {rec.get('attr','')} on {rec.get('strategy','')}"
                            elif loop == "StrategyTuner":
                                applied = "applied" if rec.get("applied") else "no gain"
                                detail = f"{rec.get('symbol','')} Sharpe {rec.get('sharpe', 0):.2f} — {applied}"
                            elif loop == "EnsembleCalibrator":
                                n = len(rec.get("changes", []))
                                detail = f"{n} weight change(s)"
                            lines.append(f"  [{ts}] {loop}: {detail}")
                        except Exception:
                            pass
            except Exception:
                pass

        lines.append("─" * 38)
        return "\n".join(lines)

    def _load_persisted_params(self):
        """On startup, restore previously optimized params into live engines."""
        if not PARAMS_FILE.exists():
            return
        try:
            stored = json.loads(PARAMS_FILE.read_text())
            for strategy_name, params in stored.items():
                engine = self.sm.engines.get(strategy_name)
                if engine is None:
                    continue
                applied = []
                for k, v in params.items():
                    if hasattr(engine, k):
                        setattr(engine, k, v)
                        applied.append(k)
                if applied:
                    self.notify(
                        f"[AutoOptimizer] Restored {len(applied)} param(s) for {strategy_name}.",
                        priority="normal",
                    )
        except Exception:
            pass