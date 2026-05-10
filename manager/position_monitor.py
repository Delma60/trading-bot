"""
manager/position_monitor.py — External close detector

Runs every 5 seconds, compares known open tickets against MT5,
and fires on_external_close() for anything that disappeared
without the bot explicitly closing it (SL hits, TP hits,
manual closes from the app, margin calls, etc.).
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
import MetaTrader5 as mt5


class PositionMonitor:

    CHECK_INTERVAL = 5  # seconds

    def __init__(self, broker, on_external_close: Callable):
        self.broker = broker
        self.on_external_close = on_external_close
        self._known: dict[int, dict] = {}      # ticket → snapshot
        self._bot_closed: set[int] = set()     # tickets the bot closed itself
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._seed()
        self._running = True
        self._thread = threading.Thread(
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
        current = self.broker.getPositions() or []
        current_tickets = {p.ticket for p in current}

        with self._lock:
            gone = set(self._known.keys()) - current_tickets

            for ticket in gone:
                meta = self._known.pop(ticket)
                if ticket in self._bot_closed:
                    self._bot_closed.discard(ticket)
                    continue                           # bot closed it, skip
                profit, close_price = self._fetch_deal(ticket)
                self.on_external_close(
                    ticket=ticket,
                    symbol=meta["symbol"],
                    profit=profit,
                    close_price=close_price,
                    direction=meta["direction"],
                    lots=meta["lots"],
                    open_price=meta["open_price"],
                )

            # Register any externally-opened positions we haven't seen before
            for p in current:
                if p.ticket not in self._known:
                    self._known[p.ticket] = self._snap(p)

    def _fetch_deal(self, ticket: int) -> tuple[float, float]:
        """Return (profit, close_price) for a closed position from MT5 history."""
        try:
            now = datetime.now()
            deals = mt5.history_deals_get(
                int((now - timedelta(days=1)).timestamp()),
                int(now.timestamp()),
            )
            if deals:
                for deal in deals:
                    if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                        return float(deal.profit), float(deal.price)
        except Exception:
            pass
        return 0.0, 0.0

    @staticmethod
    def _snap(p) -> dict:
        return {
            "symbol":     p.symbol,
            "direction":  "BUY" if p.type == 0 else "SELL",
            "lots":       float(p.volume),
            "open_price": float(p.price_open),
        }