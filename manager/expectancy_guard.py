"""
manager/expectancy_guard.py

Reads trade_history.csv and computes rolling expectancy.
Used by ProfitGuard to close (or alert on) positions whose
unrealised loss already exceeds what history justifies.

Expectancy  =  (win_rate × avg_win)  −  (loss_rate × avg_loss)

"Loss threshold" for a live trade:
    If |current_profit| > avg_loss × LOSS_MULT_ALERT  → fire alert
    If |current_profit| > avg_loss × LOSS_MULT_CLOSE  → close the trade

Both thresholds are only applied when we have at least MIN_TRADES
closed trades AND the position is already underwater (profit < 0).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable, Optional

TRADE_HISTORY = Path("data/trade_history.csv")

# How many closed trades before we trust the stats enough to act
MIN_TRADES = 20

# Rolling window — how many recent trades to average
ROLLING_WINDOW = 40

# Multipliers relative to avg_loss
LOSS_MULT_ALERT = 1.2   # notify when loss > 1.2× avg_loss
LOSS_MULT_CLOSE = 2.0   # close  when loss > 2.0× avg_loss

# How often to recalculate stats (seconds)
REFRESH_INTERVAL = 120


class ExpectancyStats:
    """Immutable snapshot of rolling expectancy statistics."""

    __slots__ = (
        "sample_size", "win_rate", "avg_win", "avg_loss",
        "expectancy", "loss_alert_threshold", "loss_close_threshold",
    )

    def __init__(
        self,
        sample_size: int,
        win_rate:    float,
        avg_win:     float,
        avg_loss:    float,     # positive number
        expectancy:  float,
    ):
        self.sample_size  = sample_size
        self.win_rate     = win_rate
        self.avg_win      = avg_win
        self.avg_loss     = avg_loss
        self.expectancy   = expectancy
        # Derived thresholds (stored as positive amounts)
        self.loss_alert_threshold = avg_loss * LOSS_MULT_ALERT
        self.loss_close_threshold = avg_loss * LOSS_MULT_CLOSE

    def __repr__(self) -> str:
        return (
            f"ExpectancyStats(n={self.sample_size}, "
            f"WR={self.win_rate:.0%}, "
            f"E=${self.expectancy:+.2f}, "
            f"avgW=${self.avg_win:.2f}, avgL=${self.avg_loss:.2f}, "
            f"alertAt=${self.loss_alert_threshold:.2f}, "
            f"closeAt=${self.loss_close_threshold:.2f})"
        )


class ExpectancyGuard:
    """
    Background service that maintains rolling expectancy statistics and
    evaluates live positions against them.

    Usage (from risk_manager.py ProfitGuard.__init__)
    --------------------------------------------------
        self._exp_guard = ExpectancyGuard(notify_callback=self.notify)
        self._exp_guard.start()

    In ProfitGuard._evaluate():
        verdict = self._exp_guard.evaluate(symbol, profit)
        if verdict == "close":
            self._close_atomic(pos, reason="Loss exceeded expectancy threshold")
        elif verdict == "alert":
            self.notify(f"⚠️ {symbol} loss ${abs(profit):.2f} exceeds avg loss …")
    """

    def __init__(self, notify_callback: Callable = print):
        self.notify       = notify_callback
        self._stats:      Optional[ExpectancyStats] = None
        self._lock        = threading.RLock()
        self._running     = False
        self._thread:     Optional[threading.Thread] = None
        # Track which tickets we've already alerted on to avoid spam
        self._alerted:    set[int] = set()
        self._closed_by_exp: set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._refresh_stats()          # immediate first load
        self._thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="ExpectancyGuard",
        )
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def stats(self) -> Optional[ExpectancyStats]:
        with self._lock:
            return self._stats

    def evaluate(self, ticket: int, profit: float) -> str:
        """
        Return one of:  "ok" | "alert" | "close"

        Only fires when:
        - profit is negative (position is underwater)
        - We have enough historical data
        - The loss exceeds the relevant threshold
        """
        if profit >= 0:
            return "ok"

        with self._lock:
            s = self._stats

        if s is None or s.sample_size < MIN_TRADES:
            return "ok"

        loss_amount = abs(profit)

        if loss_amount >= s.loss_close_threshold:
            if ticket not in self._closed_by_exp:
                self._closed_by_exp.add(ticket)
                self._alerted.discard(ticket)   # clear alert flag too
                return "close"
            return "ok"   # already actioned

        if loss_amount >= s.loss_alert_threshold:
            if ticket not in self._alerted:
                self._alerted.add(ticket)
                return "alert"

        return "ok"

    def clear_ticket(self, ticket: int):
        """Call when a position closes so we reset its tracking state."""
        self._alerted.discard(ticket)
        self._closed_by_exp.discard(ticket)

    def summary(self) -> str:
        with self._lock:
            s = self._stats
        if s is None:
            return "ExpectancyGuard: no stats yet (need closed trades)."
        return (
            f"📐 Expectancy Guard\n"
            f"   Trades sampled : {s.sample_size} (last {ROLLING_WINDOW})\n"
            f"   Win rate       : {s.win_rate:.0%}\n"
            f"   Avg win / loss : +${s.avg_win:.2f} / -${s.avg_loss:.2f}\n"
            f"   Expectancy     : ${s.expectancy:+.2f} per trade\n"
            f"   Alert at loss  : ${s.loss_alert_threshold:.2f} "
            f"({LOSS_MULT_ALERT}× avg loss)\n"
            f"   Close at loss  : ${s.loss_close_threshold:.2f} "
            f"({LOSS_MULT_CLOSE}× avg loss)"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _background_loop(self):
        while self._running:
            time.sleep(REFRESH_INTERVAL)
            try:
                self._refresh_stats()
            except Exception:
                pass

    def _refresh_stats(self):
        profits = self._load_profits()
        if len(profits) < MIN_TRADES:
            return

        recent = profits[-ROLLING_WINDOW:]
        wins   = [p for p in recent if p > 0]
        losses = [p for p in recent if p < 0]
        n      = len(recent)

        if not losses:          # all wins — nothing to guard against
            return

        win_rate = len(wins) / n
        avg_win  = sum(wins)  / len(wins)  if wins   else 0.0
        avg_loss = abs(sum(losses) / len(losses))     # positive

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        with self._lock:
            self._stats = ExpectancyStats(
                sample_size = n,
                win_rate    = win_rate,
                avg_win     = round(avg_win, 4),
                avg_loss    = round(avg_loss, 4),
                expectancy  = round(expectancy, 4),
            )

    @staticmethod
    def _load_profits() -> list[float]:
        if not TRADE_HISTORY.exists():
            return []
        profits: list[float] = []
        try:
            with open(TRADE_HISTORY, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if row.get("Action", "").upper() not in (
                        "CLOSE", "CLOSE_SL_TP", "PARTIAL_CLOSE"
                    ):
                        continue
                    raw = row.get("Profit", "")
                    try:
                        val = float(str(raw).replace("Profit:", "").strip())
                        profits.append(val)
                    except (ValueError, AttributeError):
                        pass
        except Exception:
            pass
        return profits