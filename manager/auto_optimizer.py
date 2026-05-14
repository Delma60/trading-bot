"""
manager/auto_optimizer.py — Automated Strategy Calibration & Adaptation

Coordinates three asynchronous optimization loops to tune parameters, adapt thresholds,
and calibrate ensemble priority weights against real-time market metrics safely.
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
    """Thread-safe persistence of runtime optimizer actions."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _load_closed_trades(limit: int = 500, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reads closed trades safely from the CSV history file.
    Extracts timestamp, symbol, strategy, numerical profit, and market regime.
    """
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
                    # Safely convert strings like "Profit: 12.5" to clean floats
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
    """Calculates the proportion of profitable trades inside a given subset."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if float(t.get("profit", 0.0)) > 0)
    return wins / len(trades)


def _profit_factor(trades: List[Dict[str, Any]]) -> float:
    """Calculates gross gains divided by gross losses safely."""
    gross_profit = sum(float(t.get("profit", 0.0)) for t in trades if float(t.get("profit", 0.0)) > 0)
    gross_loss   = sum(abs(float(t.get("profit", 0.0))) for t in trades if float(t.get("profit", 0.0)) < 0)
    if gross_loss == 0:
        return 99.0 if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def _atomic_write(path: Path, data: dict):
    """Safely updates target storage configurations using staging file replacements."""
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

    For each strategy, evaluates a rolling 20-trade window. If the win rate drops
    below a target floor, shifts internal tolerances to make entry screening stricter.
    Conversely, excellent performance relaxes screening parameters slightly.
    """

    INTERVAL      = 60        # seconds
    WINDOW        = 20        # trades in rolling window
    WR_TOO_LOW    = 0.45
    WR_TOO_HIGH   = 0.62

    # Strategy attribute that acts as the main quality gate
    THRESHOLD_ATTR = {
        "Mean_Reversion":  "min_rr",
        "Momentum":        "adx_min",
        "Breakout":        "volume_multiplier",
        "Scalping":        None,
        "Trend_Following": "adx_threshold",
    }

    # Nudge delta values per property (positive values enforce higher selectivity)
    NUDGE_UP_MEANS = {
        "min_rr":            +0.05,
        "adx_min":           +1.0,
        "volume_multiplier": +0.05,
        "adx_threshold":     +1.0,
    }

    # Bug 8 Fixed: Concrete min/max boundaries protecting running engine stability
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
        if len(trades) < 10:
            return

        by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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

            if not engine or not attr or not hasattr(engine, attr):
                continue

            current = float(getattr(engine, attr))
            nudge   = self.NUDGE_UP_MEANS.get(attr, 0.02)
            bounds  = self.ATTR_BOUNDS.get(attr, (0.1, 100.0))

            if wr < self.WR_TOO_LOW:
                new_val = min(current + nudge, current * 1.15)
                new_val = min(new_val, bounds[1])  # Apply absolute bounds ceiling
                setattr(engine, attr, round(new_val, 4))
                self._log("nudge_up", strat_name, attr, current, new_val, wr)

            elif wr > self.WR_TOO_HIGH:
                new_val = max(current - nudge * 0.5, current * 0.92)
                new_val = max(new_val, bounds[0])  # Apply absolute bounds floor
                setattr(engine, attr, round(new_val, 4))
                self._log("nudge_down", strat_name, attr, current, new_val, wr)

    def _log(self, direction: str, strategy: str, attr: str, old: float, new: float, wr: float):
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
    Backtest-driven numerical parameters search. Runs on a 4-hour schedule or 
    triggers reactively upon encountering sustained consecutive drawdowns on a symbol.
    """

    INTERVAL                 = 4 * 3600
    CONSECUTIVE_LOSS_TRIGGER = 8
    N_TRIALS                 = 30
    MIN_SHARPE_GAIN          = 0.10

    def __init__(self, strategy_manager, notify: Callable):
        self.sm         = strategy_manager
        self.notify     = notify
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._opt_lock  = threading.RLock()  # Upgraded to Reentrant Lock to avoid manual deadlocks
        self._last_run: Dict[str, float] = {}
        self._consec_losses: Dict[str, int] = {}

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
        """Monitors closed positions to identify drawdown cascades dynamically."""
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
        """Manually overrides execution schedules to tune a selected asset immediately."""
        if not self._opt_lock.acquire(blocking=False):
            return "Tuning already active — scheduling skipped."
        try:
            result = self._tune_symbol(symbol, "manual")
            return result or f"Tuning evaluation completed for {symbol}."
        finally:
            self._opt_lock.release()

    def _loop(self):
        time.sleep(300)  # Delay start to avoid startup content loading competition
        while self._running:
            try:
                self._scheduled_cycle()
            except Exception:
                pass
            time.sleep(self.INTERVAL)

    def _scheduled_cycle(self):
        trades = _load_closed_trades(limit=300)
        if len(trades) < 20:
            return

        by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in trades[-100:]:
            by_sym[t["symbol"]].append(t)

        worst_sym = None
        worst_wr  = 1.0
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
        with self._opt_lock:
            now = time.time()
            last = self._last_run.get(symbol, 0)
            if now - last < 3600 and reason != "manual":
                return None

            self.notify(
                f"[AutoOptimizer] Tuning {symbol} — {reason} ({self.N_TRIALS} trials)...",
                priority="normal",
            )

            try:
                try:
                    from optimizer import StrategyParamOptimizer, STRATEGY_PARAM_SPACES
                    spaces = STRATEGY_PARAM_SPACES
                except ImportError:
                    from optimizer import StrategyParamOptimizer
                    spaces = list(self.sm.engines.keys())

                trades = _load_closed_trades(symbol=symbol, limit=100)
                by_strat: Dict[str, int] = defaultdict(int)
                for t in trades:
                    s = t.get("strategy", "")
                    if s and s != "Unknown":
                        by_strat[s] += 1

                target = max(by_strat, key=by_strat.get) if by_strat else "Mean_Reversion" # type: ignore
                if target not in spaces:
                    target = next((s for s in spaces if s in self.sm.engines), "Mean_Reversion")

                opt = StrategyParamOptimizer(self.sm, notify=lambda *a, **kw: None)
                result = opt.random_search(
                    strategy_name = target,
                    symbol        = symbol,
                    n_trials      = self.N_TRIALS,
                    metric        = "sharpe",
                    notify        = lambda *a, **kw: None,
                )

                best_metric = getattr(result, "best_metric", 0.0)
                best_params = getattr(result, "best_params", {})
                applied = False

                if best_metric > self.MIN_SHARPE_GAIN and best_params:
                    engine = self.sm.engines.get(target)
                    if engine:
                        for k, v in best_params.items():
                            if hasattr(engine, k):
                                setattr(engine, k, v)
                        applied = True
                        self._persist_params(target, best_params)

                self._last_run[symbol] = now
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

                status = "✅ applied" if applied else "ℹ️ no gain"
                msg = f"[AutoOptimizer] {symbol} / {target}: Sharpe {best_metric:.2f} | {status}"
                self.notify(msg, priority="normal")
                return msg

            except Exception as exc:
                _append_log({
                    "loop": "StrategyTuner", 
                    "ts": datetime.now().isoformat(),
                    "symbol": symbol, 
                    "error": str(exc)
                })
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
    Daily rebalancing logic adjusting structural MetaScorer regime mapping weightings
    based on relative empirical success distribution paths.
    """

    SCHEDULE_HOUR  = 22
    MIN_WEIGHT     = 0.2
    MAX_WEIGHT     = 4.0
    MIN_TRADES_REQ = 5

    def __init__(self, strategy_manager, notify: Callable, learner: Optional[Any] = None):
        self.sm      = strategy_manager
        self.notify  = notify
        self.learner = learner
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: Optional[datetime] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EnsembleCalibrator")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == self.SCHEDULE_HOUR:
                    if not self._last_run or (now_utc - self._last_run).total_seconds() > 20 * 3600:
                        # Bug 1 Fixed: Resolves internal attributes directly without signature type exceptions
                        self._calibrate()
                        self._last_run = now_utc
            except Exception:
                pass
            time.sleep(600)

    def force_calibrate(self) -> str:
        """Manual execution hook wrapping background ensemble alignment evaluations."""
        return self._calibrate()

    def _calibrate(self) -> str:
        """Re-weighs empirical regime distribution scores against live outcomes."""
        trades = _load_closed_trades(limit=500)
        if len(trades) < 30:
            return "Insufficient trade history for empirical ensemble calibration."

        current_regime_fallback = (
            self.learner.get_current_regime() if self.learner and hasattr(self.learner, "get_current_regime") else "Unknown"
        )

        regime_strat: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for t in trades:
            strat  = t.get("strategy", "Unknown")
            profit = float(t.get("profit", 0.0))
            regime = t.get("regime") or current_regime_fallback
            if regime:
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
            # Safe Fallback: Direct priority scaling updates if meta_scorer module mapping is missing
            strat_profits: Dict[str, float] = defaultdict(float)
            for t in trades:
                strat_profits[t.get("strategy", "Unknown")] += float(t.get("profit", 0.0))
            
            best_strat = max(strat_profits, key=strat_profits.get) if strat_profits else None # type: ignore
            if best_strat and best_strat in self.sm.engines:
                eng = self.sm.engines[best_strat]
                eng.weight = min(eng.weight * 1.05, 2.0)
                changes.append(f"DirectPriority/{best_strat} upscaled")

        _append_log({
            "loop":    "EnsembleCalibrator",
            "ts":      datetime.now().isoformat(),
            "changes": changes,
        })

        if changes:
            return f"Ensemble calibrated successfully. {len(changes)} weight adjustments applied."
        return "Ensemble alignment check complete — current priority weights verified as stable."


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator — AutoOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class AutoOptimizer:
    """
    Coordinates execution lifecycles across the background tuning, threshold adaptation,
    and meta-scoring calibration subsystems.
    """

    def __init__(self, strategy_manager, notify_callback: Callable = print, learner: Optional[Any] = None):
        self.sm      = strategy_manager
        self.notify  = notify_callback
        self.learner = learner

        self.micro      = MicroAdaptor(strategy_manager, notify_callback)
        self.tuner      = StrategyTuner(strategy_manager, notify_callback)
        self.calibrator = EnsembleCalibrator(strategy_manager, notify_callback, learner)

        self._load_persisted_params()

    def start(self):
        """Initiates parallel tuning threads securely."""
        self.micro.start()
        self.tuner.start()
        self.calibrator.start()
        self.notify(
            "[AutoOptimizer] Parallel optimization engine online: MicroAdaptor (60s), "
            "StrategyTuner (4h), EnsembleCalibrator (Daily 22:00 UTC).",
            priority="normal",
        )

    def stop(self):
        """Safely terminates all operational optimization tasks."""
        self.micro.stop()
        self.tuner.stop()
        self.calibrator.stop()

    def on_trade_closed(self, symbol: str, profit: float):
        """Event dispatch forwarding state outcomes to localized memory engines."""
        self.tuner.record_outcome(symbol, profit)

    def force_tune(self, symbol: Optional[str] = None) -> str:
        """Forces direct numerical search optimization execution loops."""
        if symbol is None:
            trades = _load_closed_trades(limit=200)
            by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for t in trades[-100:]:
                by_sym[t["symbol"]].append(t)
            symbol = min(by_sym, key=lambda s: _win_rate(by_sym[s])) if by_sym else "EURUSD" # type: ignore
        return self.tuner.force_tune(symbol)

    def force_calibrate(self) -> str:
        """Forces immediate structural ensemble balancing routines."""
        return self.calibrator.force_calibrate()

    def report(self) -> str:
        """Assembles structured analytical breakdowns summarizing ongoing system behaviors."""
        trades = _load_closed_trades(limit=300)
        if not trades:
            return "Optimization analytics pending — data ingestion active awaiting initial closures."

        recent = trades[-50:]
        overall_wr = _win_rate(recent)
        overall_pf = _profit_factor(recent)

        by_strat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_sym:   Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in recent:
            s = t.get("strategy", "Unknown")
            if s and s != "Unknown":
                by_strat[s].append(t)
            by_sym[t["symbol"]].append(t)

        lines = [
            "🤖 AutoOptimizer — System Status Dashboard",
            "─" * 45,
            f"Global Outcomes (Last {len(recent)} entries): WR {overall_wr:.0%} | PF {overall_pf:.2f}",
            "",
            "Strategy Evaluation:",
        ]
        
        for s, ts in sorted(by_strat.items(), key=lambda x: _win_rate(x[1])):
            wr  = _win_rate(ts)
            pf  = _profit_factor(ts)
            eng = self.sm.engines.get(s)
            
            gate = ""
            for attr in ("adx_min", "adx_threshold", "min_rr", "volume_multiplier"):
                if eng and hasattr(eng, attr):
                    gate = f" [{attr}={getattr(eng, attr):.2f}]"
                    break
                    
            flag = " ⚠️" if wr < 0.45 else " ✅"
            lines.append(f"  {s:<20} WR {wr:.0%} PF {pf:.2f}{gate}{flag}")

        lines.append("")
        lines.append("Asset Profile (Highest Vulnerability):")
        sym_sorted = sorted(by_sym.items(), key=lambda x: _win_rate(x[1]))
        for sym, ts in sym_sorted[:5]:
            wr = _win_rate(ts)
            flag = " ⚠️" if wr < 0.45 else " ✅"
            lines.append(f"  {sym:<12} WR {wr:.0%} ({len(ts)} total positions){flag}")

        if LOG_FILE.exists():
            try:
                log_lines = LOG_FILE.read_text().strip().splitlines()[-5:]
                if log_lines:
                    lines += ["", "System History Logging:"]
                    for ll in log_lines:
                        try:
                            rec = json.loads(ll)
                            loop = rec.get("loop", "?")
                            ts   = rec.get("ts", "")[:16]
                            detail = ""
                            if loop == "MicroAdaptor":
                                detail = f"{rec.get('direction','')} {rec.get('attr','')} on {rec.get('strategy','')}"
                            elif loop == "StrategyTuner":
                                applied = "applied" if rec.get("applied") else "no updates"
                                detail = f"{rec.get('symbol','')} Sharpe {rec.get('sharpe', 0):.2f} — {applied}"
                            elif loop == "EnsembleCalibrator":
                                detail = f"{len(rec.get('changes', []))} balancing modification(s)"
                            lines.append(f"  [{ts}] {loop}: {detail}")
                        except Exception:
                            pass
            except Exception:
                pass

        lines.append("─" * 45)
        return "\n".join(lines)

    def _load_persisted_params(self):
        """Restores historically optimized functional profiles smoothly on initialization."""
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
                        f"[AutoOptimizer] Initialized {len(applied)} persistent configuration parameter(s) on {strategy_name}.",
                        priority="normal",
                    )
        except Exception:
            pass