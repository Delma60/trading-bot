"""
manager/auto_optimizer.py — Automated Strategy Calibration & Adaptation

Combines an EnsembleCalibrator to optimize voting weights against live market metrics
with a background MicroAdaptor to tune running engine configuration tolerances safely.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_PATH = Path("data/auto_optimizer_log.jsonl")


def _append_log(record: dict):
    """Thread-safe persistence of runtime optimizer actions."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


class EnsembleCalibrator:
    """
    Evaluates running voting distributions against trailing asset execution history.
    Automatically assigns optimal scaling adjustments to the underlying engines.
    """

    def __init__(self, broker, learner, strategy_manager, interval_hours: int = 12):
        self.broker = broker
        self.learner = learner
        self.strategy_manager = strategy_manager
        self.interval_seconds = interval_hours * 3600
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="EnsembleCalibrator"
        )
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self.running:
            time.sleep(self.interval_seconds)
            try:
                # Bug 1 Fixed: Explicitly inject structural framework dependencies
                self._calibrate(self.learner, self.strategy_manager)
            except Exception as e:
                _append_log({"event": "calibrate_loop_error", "error": str(e)})

    def force_calibrate(self) -> str:
        """Expose manual trigger framework directly to user messaging channels."""
        # Bug 1 Fixed: Propagate framework objects cleanly during manual invocations
        return self._calibrate(self.learner, self.strategy_manager)

    def _calibrate(self, learner, strategy_manager) -> str:
        """Resolves optimal historical regime matching across available active weights."""
        if not learner or not strategy_manager:
            return "Dependencies unready. Calibration skipped."

        history_path = Path("data/trade_history.json")
        if not history_path.exists():
            return "No baseline history recorded. Ensemble weights uncalibrated."

        try:
            with history_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return "Corrupted historical structure. Calibration failed."

        if len(records) < 20:
            return f"Insufficient execution history ({len(records)}/20). Baseline preserved."

        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in records:
            strat = r.get("strategy")
            prof = r.get("profit", 0.0)
            if strat and strat in strategy_manager.engines:
                scores[strat] = scores.get(strat, 0.0) + prof
                counts[strat] = counts.get(strat, 0) + 1

        if not scores:
            return "No direct performance matches identified across loaded engines."

        best_strat = max(scores, key=lambda k: scores[k] if counts[k] > 5 else -999) # type: ignore
        if scores[best_strat] <= 0:
            return "Highest tracking strategies currently underwater. Retaining standard baselines."

        for name, engine in strategy_manager.engines.items():
            if name == best_strat:
                engine.weight = min(engine.weight * 1.10, 2.0)
            else:
                engine.weight = max(engine.weight * 0.95, 0.5)

        _append_log({
            "timestamp": datetime.now().isoformat(),
            "event": "ensemble_calibrated",
            "promoted": best_strat,
            "weights": {n: round(e.weight, 2) for n, e in strategy_manager.engines.items()}
        })
        return f"Ensemble updated successfully. Engine priority anchored to {best_strat}."


class MicroAdaptor:
    """
    Periodically nudges operational tolerance attributes based on localized performance feedback.
    Enforces floor and ceiling configurations to prevent operational parameter starvation.
    """

    FLOOR = 0.50
    CEILING = 0.85

    def __init__(self, strategy_manager, interval_minutes: int = 15):
        self.strategy_manager = strategy_manager
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._cycle, daemon=True, name="MicroAdaptor")
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _cycle(self):
        while self.running:
            time.sleep(self.interval_seconds)
            try:
                for name, engine in self.strategy_manager.engines.items():
                    # Focus background adjustments onto trending framework properties
                    if hasattr(engine, "adx_threshold"):
                        current = getattr(engine, "adx_threshold")
                        # Emulate functional drag across volatile indicators
                        nudge = -0.5 if engine.weight > 1.2 else 0.5
                        raw_val = current + nudge
                        
                        # Bug 8 Fixed: Apply operational floor/ceiling bounding constraints
                        bounded_val = max(20.0 * self.FLOOR, min(raw_val, 40.0 * self.CEILING))
                        setattr(engine, "adx_threshold", round(bounded_val, 2))
            except Exception:
                pass