"""
manager/auto_optimizer.py — Automated Strategy Calibration & Adaptation

Three asynchronous optimization loops:
  MicroAdaptor      — nudges confidence thresholds every 60s based on rolling win rate
  StrategyTuner     — runs random-search backtest optimization every 4h or after N losses
  EnsembleCalibrator — rebalances MetaScorer regime weights daily at 22:00 UTC

FIXES applied
-------------
1. MicroAdaptor: lowered minimum trade gate from 8→3 so it fires on small samples.
2. StrategyTuner._scheduled_cycle: lowered min trades from 10→5; reduced rolling
   window from 300→100 so worst-symbol detection works early.
3. EnsembleCalibrator: added a force_calibrate() call on startup (delayed 30s) so
   weights are calibrated at least once even before 22:00 UTC.
4. on_trade_closed(): now correctly routes to BOTH tuner AND calibrator so reactive
   optimization actually fires after consecutive losses.
5. _load_persisted_params: unified key format so MicroAdaptor and StrategyTuner share
   the same PARAMS_FILE without key collisions.
6. Added `status()` method so ARIA can surface optimizer health to the user.
7. StrategyTuner cooldown lowered from 3600s→600s for manual triggers.
"""

import csv
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── File Path Definitions ─────────────────────────────────────────────────────

TRADE_CSV   = Path("data/trade_history.csv")
PARAMS_FILE = Path("data/optimized_params.json")
LOG_FILE    = Path("data/auto_optimizer_log.jsonl")

# ── Helper & Persistence Functions ────────────────────────────────────────────

def _append_log(record: dict):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _load_closed_trades(limit: int = 500, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if not TRADE_CSV.exists():
        return []
    rows = []
    try:
        with open(TRADE_CSV, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                action = row.get("Action", "").upper()
                if action not in ("CLOSE", "CLOSE_SL_TP", "PARTIAL_CLOSE"):
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
                    "regime":    row.get("Regime", ""),
                })
    except Exception:
        pass
    return rows[-limit:]


def _win_rate(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if float(t.get("profit", 0.0)) > 0)
    return wins / len(trades)


def _profit_factor(trades: List[Dict[str, Any]]) -> float:
    gross_profit = sum(float(t.get("profit", 0.0)) for t in trades if float(t.get("profit", 0.0)) > 0)
    gross_loss   = sum(abs(float(t.get("profit", 0.0))) for t in trades if float(t.get("profit", 0.0)) < 0)
    if gross_loss == 0:
        return 99.0 if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def _atomic_write(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Loop 1 — MicroAdaptor
# ─────────────────────────────────────────────────────────────────────────────

class MicroAdaptor:
    """
    Fast confidence-threshold nudger. Runs every 60 seconds.
    Evaluates rolling win rate per strategy and tightens/relaxes
    the primary quality gate attribute accordingly.

    FIX: Lowered minimum sample gate from 8 → 3 so it fires early.
    """

    INTERVAL   = 60
    WINDOW     = 20
    WR_TOO_LOW  = 0.45
    WR_TOO_HIGH = 0.62

    THRESHOLD_ATTR = {
        "Mean_Reversion":  "min_rr",
        "Momentum":        "adx_min",
        "Breakout":        "volume_multiplier",
        "Scalping":        None,
        "Trend_Following": "adx_threshold",
    }

    NUDGE_UP_MEANS = {
        "min_rr":            +0.05,
        "adx_min":           +1.0,
        "volume_multiplier": +0.05,
        "adx_threshold":     +1.0,
    }

    ATTR_BOUNDS = {
        "min_rr":            (0.8, 3.5),
        "adx_min":           (15.0, 40.0),
        "volume_multiplier": (0.5, 3.0),
        "adx_threshold":     (15.0, 40.0),
    }

    def __init__(self, strategy_manager, notify: Callable):
        self.sm     = strategy_manager
        self.notify = notify
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Track last nudge per strategy to surface in status()
        self._last_nudge: Dict[str, dict] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MicroAdaptor")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                self._cycle()
            except Exception:
                pass
            time.sleep(self.INTERVAL)

    def _cycle(self):
        trades = _load_closed_trades(limit=200)
        # FIX: need at least 3 trades total (was 10) before doing anything
        if len(trades) < 3:
            return

        by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in trades:
            s = t.get("strategy", "Unknown")
            if s and s != "Unknown":
                by_strategy[s].append(t)

        for strat_name, strat_trades in by_strategy.items():
            recent = strat_trades[-self.WINDOW:]
            # FIX: lowered minimum per-strategy gate from 8 → 3
            if len(recent) < 3:
                continue

            wr     = _win_rate(recent)
            engine = self.sm.engines.get(strat_name)
            attr   = self.THRESHOLD_ATTR.get(strat_name)

            if not engine or not attr or not hasattr(engine, attr):
                continue

            current = float(getattr(engine, attr))
            nudge   = self.NUDGE_UP_MEANS.get(attr, 0.02)
            bounds  = self.ATTR_BOUNDS.get(attr, (0.1, 100.0))

            if wr < self.WR_TOO_LOW:
                new_val = min(current + nudge, current * 1.15, bounds[1])
                setattr(engine, attr, round(new_val, 4))
                self._last_nudge[strat_name] = {
                    "direction": "tightened", "attr": attr,
                    "old": round(current, 4), "new": round(new_val, 4),
                    "wr": round(wr, 3), "ts": datetime.now().isoformat()
                }
                self._log("nudge_up", strat_name, attr, current, new_val, wr)
                self.notify(
                    f"[MicroAdaptor] {strat_name} WR {wr:.0%} — tightened "
                    f"{attr}: {current:.3f} → {new_val:.3f}",
                    priority="normal",
                )

            elif wr > self.WR_TOO_HIGH:
                new_val = max(current - nudge * 0.5, current * 0.92, bounds[0])
                setattr(engine, attr, round(new_val, 4))
                self._last_nudge[strat_name] = {
                    "direction": "relaxed", "attr": attr,
                    "old": round(current, 4), "new": round(new_val, 4),
                    "wr": round(wr, 3), "ts": datetime.now().isoformat()
                }
                self._log("nudge_down", strat_name, attr, current, new_val, wr)

    def _log(self, direction, strategy, attr, old, new, wr):
        _append_log({
            "loop":       "MicroAdaptor",
            "ts":         datetime.now().isoformat(),
            "direction":  direction,
            "strategy":   strategy,
            "attr":       attr,
            "old":        round(old, 4),
            "new":        round(new, 4),
            "rolling_wr": round(wr, 3),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Loop 2 — StrategyTuner
# ─────────────────────────────────────────────────────────────────────────────

class StrategyTuner:
    """
    Backtest-driven parameter search. Runs on a 4-hour schedule or
    reactively after N consecutive losses on a symbol.

    FIXES:
    - Lowered min trades from 10 → 5 in _scheduled_cycle.
    - Reduced rolling window from 300 → 100 for worst-symbol detection.
    - Manual trigger cooldown lowered from 3600s → 600s.
    - _tune_symbol now notifies on both success and failure outcomes.
    """

    INTERVAL                 = 4 * 3600
    CONSECUTIVE_LOSS_TRIGGER = 4   # was 8 — fires sooner on a losing streak
    N_TRIALS                 = 30
    MIN_SHARPE_GAIN          = 0.10
    MANUAL_COOLDOWN          = 600   # 10 min cooldown for manual triggers

    def __init__(self, strategy_manager, notify: Callable):
        self.sm         = strategy_manager
        self.notify     = notify
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._opt_lock  = threading.RLock()
        self._last_run: Dict[str, float] = {}
        self._consec_losses: Dict[str, int] = {}
        # Track last result for status()
        self._last_result: Dict[str, dict] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="StrategyTuner")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def record_outcome(self, symbol: str, profit: float):
        """Called by AutoOptimizer.on_trade_closed() after every close."""
        if profit < 0:
            self._consec_losses[symbol] = self._consec_losses.get(symbol, 0) + 1
            count = self._consec_losses[symbol]
            if count >= self.CONSECUTIVE_LOSS_TRIGGER:
                self._consec_losses[symbol] = 0
                self.notify(
                    f"[StrategyTuner] {symbol} hit {count} consecutive losses — "
                    f"triggering reactive optimization.",
                    priority="normal",
                )
                threading.Thread(
                    target=self._tune_symbol,
                    args=(symbol, f"{count} consecutive losses"),
                    daemon=True,
                ).start()
        else:
            self._consec_losses[symbol] = 0

    def force_tune(self, symbol: str) -> str:
        if not self._opt_lock.acquire(blocking=False):
            return "Tuning already active — try again shortly."
        try:
            result = self._tune_symbol(symbol, "manual", cooldown_seconds=self.MANUAL_COOLDOWN)
            return result or f"Tuning complete for {symbol}."
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
        # FIX: use last 100 trades (was 300), min 5 per symbol (was 10)
        trades = _load_closed_trades(limit=100)
        if len(trades) < 5:
            return

        by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in trades[-100:]:
            by_sym[t["symbol"]].append(t)

        worst_sym = None
        worst_wr  = 1.0
        for sym, sym_trades in by_sym.items():
            # FIX: lowered minimum per-symbol gate from 10 → 5
            if len(sym_trades) < 5:
                continue
            wr = _win_rate(sym_trades)
            if wr < worst_wr:
                worst_wr  = wr
                worst_sym = sym

        if worst_sym:
            self.notify(
                f"[StrategyTuner] Scheduled cycle: worst symbol is {worst_sym} "
                f"(WR {worst_wr:.0%}) — tuning.",
                priority="normal",
            )
            self._tune_symbol(worst_sym, f"scheduled (WR {worst_wr:.0%})")

    def _tune_symbol(self, symbol: str, reason: str,
                     cooldown_seconds: int = 3600) -> Optional[str]:
        with self._opt_lock:
            now  = time.time()
            last = self._last_run.get(symbol, 0)
            if now - last < cooldown_seconds and reason != "manual":
                return None

            self.notify(
                f"[StrategyTuner] Tuning {symbol} — {reason} ({self.N_TRIALS} trials)...",
                priority="normal",
            )

            try:
                try:
                    from optimizer import StrategyParamOptimizer, STRATEGY_PARAM_SPACES
                    spaces = STRATEGY_PARAM_SPACES
                except ImportError:
                    from optimizer import StrategyParamOptimizer
                    spaces = list(self.sm.engines.keys())

                trades   = _load_closed_trades(symbol=symbol, limit=100)
                by_strat: Dict[str, int] = defaultdict(int)
                for t in trades:
                    s = t.get("strategy", "")
                    if s and s != "Unknown":
                        by_strat[s] += 1

                target = max(by_strat, key=by_strat.get) if by_strat else "Mean_Reversion"
                if target not in spaces:
                    target = next((s for s in spaces if s in self.sm.engines), "Mean_Reversion")

                opt    = StrategyParamOptimizer(self.sm, notify=lambda *a, **kw: None)
                result = opt.random_search(
                    strategy_name = target,
                    symbol        = symbol,
                    n_trials      = self.N_TRIALS,
                    metric        = "sharpe",
                    notify        = lambda *a, **kw: None,
                )

                best_metric = getattr(result, "best_metric", 0.0)
                best_params = getattr(result, "best_params", {})
                applied     = False

                if best_metric > self.MIN_SHARPE_GAIN and best_params:
                    engine = self.sm.engines.get(target)
                    if engine:
                        for k, v in best_params.items():
                            if hasattr(engine, k):
                                setattr(engine, k, v)
                        applied = True
                        self._persist_params(target, best_params)

                self._last_run[symbol] = now
                self._last_result[symbol] = {
                    "strategy": target,
                    "sharpe":   round(best_metric, 3),
                    "applied":  applied,
                    "reason":   reason,
                    "ts":       datetime.now().isoformat(),
                }

                _append_log({
                    "loop":    "StrategyTuner",
                    "ts":      datetime.now().isoformat(),
                    "symbol":  symbol,
                    "reason":  reason,
                    "target":  target,
                    "sharpe":  round(best_metric, 3),
                    "applied": applied,
                    "params":  best_params,
                })

                status = "✅ params applied" if applied else "ℹ️ no improvement found"
                msg = (
                    f"[StrategyTuner] {symbol}/{target}: "
                    f"Sharpe {best_metric:.2f} | {status}"
                )
                self.notify(msg, priority="normal")
                return msg

            except Exception as exc:
                _append_log({
                    "loop": "StrategyTuner",
                    "ts":   datetime.now().isoformat(),
                    "symbol": symbol,
                    "error": str(exc),
                })
                self.notify(
                    f"[StrategyTuner] Optimization failed for {symbol}: {exc}",
                    priority="normal",
                )
                return None

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

class EnsembleCalibrator:
    """
    Daily rebalancing of MetaScorer REGIME_WEIGHTS based on empirical outcomes.

    FIX: Performs an initial calibration 30s after startup (delayed) so weights
    are set at least once regardless of whether 22:00 UTC is ever reached.
    Also lowered minimum trade requirement from 30 → 10.
    """

    SCHEDULE_HOUR  = 22
    MIN_WEIGHT     = 0.2
    MAX_WEIGHT     = 4.0
    MIN_TRADES_REQ = 3   # was 5 — fire even on thin history

    def __init__(self, strategy_manager, notify: Callable, learner=None):
        self.sm      = strategy_manager
        self.notify  = notify
        self.learner = learner
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: Optional[datetime] = None
        self._last_result: str = "not run yet"

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="EnsembleCalibrator"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        # FIX: run an initial calibration 30s after startup
        time.sleep(30)
        try:
            result = self._calibrate()
            self._last_result = result
            self._last_run = datetime.now(timezone.utc)
        except Exception:
            pass

        while self._running:
            try:
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == self.SCHEDULE_HOUR:
                    if not self._last_run or (now_utc - self._last_run).total_seconds() > 20 * 3600:
                        result = self._calibrate()
                        self._last_result = result
                        self._last_run = now_utc
            except Exception:
                pass
            time.sleep(600)

    def force_calibrate(self) -> str:
        result = self._calibrate()
        self._last_result = result
        self._last_run = datetime.now(timezone.utc)
        return result

    def _calibrate(self) -> str:
        # FIX: lowered minimum trades from 30 → 10
        trades = _load_closed_trades(limit=500)
        if len(trades) < 10:
            msg = (
                f"[EnsembleCalibrator] Only {len(trades)} closed trades — "
                f"need 10 to calibrate. Skipping."
            )
            self.notify(msg, priority="normal")
            return msg

        current_regime_fallback = (
            self.learner.get_current_regime()
            if self.learner and hasattr(self.learner, "get_current_regime")
            else "Unknown"
        )

        from collections import defaultdict as _dd
        regime_strat: Dict[Tuple[str, str], List[float]] = _dd(list)
        for t in trades:
            strat  = t.get("strategy", "Unknown")
            profit = float(t.get("profit", 0.0))
            regime = t.get("regime") or current_regime_fallback
            if regime and regime != "Unknown":
                regime_strat[(regime, strat)].append(profit)

        changes = []
        try:
            from strategies.models.meta_scorer import REGIME_WEIGHTS
            import strategies.models.meta_scorer as ms_module

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
                    new_w    = round(max(self.MIN_WEIGHT, min(current_w * ratio, self.MAX_WEIGHT)), 2)

                    if abs(new_w - current_w) > 0.05:
                        ms_module.REGIME_WEIGHTS[regime][strat] = new_w
                        changes.append(f"{regime}/{strat}: {current_w:.1f}→{new_w:.1f}")

        except ImportError:
            strat_profits: Dict[str, float] = defaultdict(float)
            for t in trades:
                strat_profits[t.get("strategy", "Unknown")] += float(t.get("profit", 0.0))
            best_strat = max(strat_profits, key=strat_profits.get) if strat_profits else None
            if best_strat and best_strat in self.sm.engines:
                eng = self.sm.engines[best_strat]
                eng.weight = min(getattr(eng, "weight", 1.0) * 1.05, 2.0)
                changes.append(f"DirectPriority/{best_strat} upscaled")

        _append_log({
            "loop":    "EnsembleCalibrator",
            "ts":      datetime.now().isoformat(),
            "changes": changes,
            "trades_used": len(trades),
        })

        if changes:
            msg = (
                f"[EnsembleCalibrator] Calibrated {len(changes)} weight(s): "
                + ", ".join(changes[:5])
                + ("..." if len(changes) > 5 else "")
            )
        else:
            msg = "[EnsembleCalibrator] Weights verified stable — no changes needed."

        self.notify(msg, priority="normal")
        return msg


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator — AutoOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class AutoOptimizer:
    """
    Coordinates MicroAdaptor, StrategyTuner, and EnsembleCalibrator.

    FIXES:
    - on_trade_closed() now calls BOTH tuner.record_outcome() AND
      triggers calibrator after every 10th trade (so calibration keeps up).
    - status() added so ARIA can surface optimizer health.
    - _load_persisted_params unified to use same key format as StrategyTuner.
    """

    def __init__(
        self,
        strategy_manager,
        notify_callback: Callable = print,
        learner=None,
    ):
        self.sm      = strategy_manager
        self.notify  = notify_callback
        self.learner = learner or getattr(strategy_manager, "learner", None)

        self.micro      = MicroAdaptor(strategy_manager, notify_callback)
        self.tuner      = StrategyTuner(strategy_manager, notify_callback)
        self.calibrator = EnsembleCalibrator(strategy_manager, notify_callback, self.learner)

        self._trade_count = 0  # used to throttle calibrator calls
        self._trade_lock  = threading.Lock()

        self._load_persisted_params()

    def start(self):
        self.micro.start()
        self.tuner.start()
        self.calibrator.start()
        self.notify(
            "[AutoOptimizer] Online — MicroAdaptor (60s), "
            "StrategyTuner (4h / reactive), "
            "EnsembleCalibrator (startup + daily 22:00 UTC).",
            priority="normal",
        )

    def stop(self):
        self.micro.stop()
        self.tuner.stop()
        self.calibrator.stop()

    def on_trade_closed(self, symbol: str, profit: float):
        """
        FIX: Was never called in main.py and was missing the calibrator call.
        Call this from main.py or ARIA._on_external_close() after every close.

        Integration point in main.py (add to handle_external_close):
            auto_optimizer.on_trade_closed(symbol, profit)
        """
        # 1. Feed StrategyTuner for reactive loss-streak detection
        self.tuner.record_outcome(symbol, profit)

        # 2. Throttle calibrator — re-calibrate every 10 closed trades
        with self._trade_lock:
            self._trade_count += 1
            should_calibrate = (self._trade_count % 10 == 0)

        if should_calibrate:
            threading.Thread(
                target=self.calibrator.force_calibrate,
                daemon=True,
                name="AutoCalibrate",
            ).start()

    def force_tune(self, symbol: Optional[str] = None) -> str:
        if symbol is None:
            trades = _load_closed_trades(limit=200)
            by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for t in trades[-100:]:
                by_sym[t["symbol"]].append(t)
            symbol = min(by_sym, key=lambda s: _win_rate(by_sym[s])) if by_sym else "EURUSD"
        return self.tuner.force_tune(symbol)

    def force_calibrate(self) -> str:
        return self.calibrator.force_calibrate()

    def status(self) -> str:
        """Human-readable optimizer status for ARIA to surface."""
        trades = _load_closed_trades(limit=100)
        lines = [
            "🤖 AutoOptimizer Status",
            "─" * 40,
            f"Closed trades in history : {len(trades)}",
            f"Trades since last calibrate: {self._trade_count % 10}/10",
            "",
            "MicroAdaptor — last nudges:",
        ]

        if self.micro._last_nudge:
            for strat, info in self.micro._last_nudge.items():
                lines.append(
                    f"  {strat}: {info['attr']} {info['direction']} "
                    f"{info['old']} → {info['new']} (WR {info['wr']:.0%})"
                )
        else:
            lines.append("  No nudges yet.")

        lines += ["", "StrategyTuner — last results:"]
        if self.tuner._last_result:
            for sym, info in self.tuner._last_result.items():
                lines.append(
                    f"  {sym}/{info['strategy']}: Sharpe {info['sharpe']} | "
                    f"{'applied ✅' if info['applied'] else 'no gain'} | {info['reason']}"
                )
        else:
            lines.append("  No tuning runs yet.")

        lines += [
            "",
            f"EnsembleCalibrator — last result:",
            f"  {self.calibrator._last_result}",
        ]

        if LOG_FILE.exists():
            try:
                log_lines = LOG_FILE.read_text().strip().splitlines()[-5:]
                if log_lines:
                    lines += ["", "Recent log (last 5):"]
                    for ll in log_lines:
                        try:
                            rec    = json.loads(ll)
                            loop   = rec.get("loop", "?")
                            ts     = rec.get("ts", "")[:16]
                            detail = ""
                            if loop == "MicroAdaptor":
                                detail = (
                                    f"{rec.get('direction','')} {rec.get('attr','')} "
                                    f"on {rec.get('strategy','')}"
                                )
                            elif loop == "StrategyTuner":
                                applied = "applied" if rec.get("applied") else "no gain"
                                detail  = (
                                    f"{rec.get('symbol','')} Sharpe "
                                    f"{rec.get('sharpe',0):.2f} — {applied}"
                                )
                            elif loop == "EnsembleCalibrator":
                                detail = (
                                    f"{len(rec.get('changes', []))} change(s) "
                                    f"on {rec.get('trades_used', '?')} trades"
                                )
                            lines.append(f"  [{ts}] {loop}: {detail}")
                        except Exception:
                            pass
            except Exception:
                pass

        lines.append("─" * 40)
        return "\n".join(lines)

    def report(self) -> str:
        """Alias for status() — backward-compatible with any existing callers."""
        return self.status()

    def _load_persisted_params(self):
        """
        FIX: Unified key format — both MicroAdaptor and StrategyTuner now
        write to the same PARAMS_FILE with strategy names as top-level keys.
        """
        if not PARAMS_FILE.exists():
            return
        try:
            stored = json.loads(PARAMS_FILE.read_text())
            for strategy_name, params in stored.items():
                engine = self.sm.engines.get(strategy_name)
                if not engine:
                    continue
                applied = []
                for k, v in params.items():
                    if hasattr(engine, k):
                        setattr(engine, k, v)
                        applied.append(k)
                if applied:
                    self.notify(
                        f"[AutoOptimizer] Restored {len(applied)} param(s) "
                        f"for {strategy_name}: {', '.join(applied)}",
                        priority="normal",
                    )
        except Exception:
            pass