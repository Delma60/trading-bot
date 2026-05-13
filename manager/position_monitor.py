"""
manager/position_monitor.py — External close detector

Runs every 5 seconds, compares known open tickets against MT5,
and fires on_external_close() for anything that disappeared
without the bot explicitly closing it (SL hits, TP hits,
manual closes from the app, margin calls, etc.).

FIX: Store live profit + price in snapshot so external closes
     always log real numbers even when MT5 deal history isn't
     indexed yet. _fetch_deal retries with backoff and a wider
     3-day lookback window as a secondary safety net.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import MetaTrader5 as mt5


class PositionMonitor:

    CHECK_INTERVAL   = 5    # seconds between reconcile sweeps
    DEAL_MAX_RETRIES = 4    # how many times to retry _fetch_deal
    DEAL_RETRY_DELAY = 0.5  # seconds between retries
    DEAL_LOOKBACK_DAYS = 3  # days of history to search

    def __init__(self, broker, on_external_close: Callable):
        self.broker            = broker
        self.on_external_close = on_external_close
        self._known:       dict[int, dict] = {}   # ticket → snapshot
        self._bot_closed:  set[int]        = set()
        self._running      = False
        self._thread:      Optional[threading.Thread] = None
        self._lock         = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._seed()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="PositionMonitor"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def mark_bot_closed(self, ticket: int):
        """Call after the bot successfully closes a position to suppress the callback."""
        with self._lock:
            self._bot_closed.add(ticket)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _seed(self):
        positions = self.broker.getPositions() or []
        with self._lock:
            for p in positions:
                self._known[p.ticket] = self._snap(p)

    def _loop(self):
        while self._running:
            try:
                if self.broker.connected:
                    self._reconcile()
            except Exception:
                pass
            time.sleep(self.CHECK_INTERVAL)

    def _reconcile(self):
        current   = self.broker.getPositions() or []
        current_map = {p.ticket: p for p in current}

        with self._lock:
            # Update live profit/price for positions still open
            for ticket, snap in self._known.items():
                if ticket in current_map:
                    p = current_map[ticket]
                    snap["last_profit"]        = float(p.profit)
                    snap["last_price_current"] = float(p.price_current)

            gone = set(self._known.keys()) - set(current_map.keys())

            for ticket in gone:
                meta = self._known.pop(ticket)

                if ticket in self._bot_closed:
                    self._bot_closed.discard(ticket)
                    continue  # bot closed it — skip

                # Try to get real numbers from MT5 deal history first.
                # Fall back to last known snapshot values if that fails.
                profit, close_price = self._fetch_deal_with_retry(ticket)

                if profit == 0.0 and close_price == 0.0:
                    # Use snapshot fallback so we never log 0.0 / 0.0
                    profit      = meta.get("last_profit",        0.0)
                    close_price = meta.get("last_price_current", meta["open_price"])

                self.on_external_close(
                    ticket      = ticket,
                    symbol      = meta["symbol"],
                    profit      = profit,
                    close_price = close_price,
                    direction   = meta["direction"],
                    lots        = meta["lots"],
                    open_price  = meta["open_price"],
                )

            # Register externally-opened positions we haven't seen before
            for ticket, p in current_map.items():
                if ticket not in self._known:
                    self._known[ticket] = self._snap(p)

    # ── Deal history lookup ───────────────────────────────────────────────────

    def _fetch_deal_with_retry(self, ticket: int) -> tuple[float, float]:
        """
        Retry _fetch_deal up to DEAL_MAX_RETRIES times with a short delay.
        MT5 deal history is not always indexed immediately after a close.
        """
        for attempt in range(self.DEAL_MAX_RETRIES):
            profit, price = self._fetch_deal(ticket)
            if profit != 0.0 or price != 0.0:
                return profit, price
            if attempt < self.DEAL_MAX_RETRIES - 1:
                time.sleep(self.DEAL_RETRY_DELAY)
        return 0.0, 0.0

    def _fetch_deal(self, ticket: int) -> tuple[float, float]:
        """
        Return (profit, close_price) for a closed position from MT5 history.

        Searches the last DEAL_LOOKBACK_DAYS days.  Tries matching on
        position_id first (standard), then falls back to matching on the
        deal ticket itself (some brokers differ).
        """
        try:
            now   = datetime.now()
            start = int((now - timedelta(days=self.DEAL_LOOKBACK_DAYS)).timestamp())
            end   = int(now.timestamp())

            deals = mt5.history_deals_get(start, end)
            if not deals:
                return 0.0, 0.0

            # Pass 1: match by position_id (the normal MT5 way)
            for deal in deals:
                if (deal.position_id == ticket
                        and deal.entry == mt5.DEAL_ENTRY_OUT):
                    return float(deal.profit), float(deal.price)

            # Pass 2: some brokers use deal.ticket == position ticket
            for deal in deals:
                if (deal.ticket == ticket
                        and deal.entry == mt5.DEAL_ENTRY_OUT):
                    return float(deal.profit), float(deal.price)

        except Exception:
            pass

        return 0.0, 0.0

    # ── Snapshot ──────────────────────────────────────────────────────────────

    @staticmethod
    def _snap(p) -> dict:
        return {
            "symbol":             p.symbol,
            "direction":          "BUY" if p.type == 0 else "SELL",
            "lots":               float(p.volume),
            "open_price":         float(p.price_open),
            # Live values updated every CHECK_INTERVAL — used as fallback
            "last_profit":        float(p.profit),
            "last_price_current": float(p.price_current),
        }