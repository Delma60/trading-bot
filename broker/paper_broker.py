"""
broker/paper_broker.py — Paper Trading Adapter

Simulates a live broker with in-memory positions and realistic fills.
Useful for:
  - Development without a live MT5 session
  - Backtesting integrations that need a broker-like API
  - Running the full ARIA stack on a machine without MT5 installed

Fills are immediate at the current simulated price. Price is either
provided by a price_feed callback or frozen at the last set_price() call.

Usage
-----
    from broker.paper_broker import PaperBroker

    paper = PaperBroker(initial_balance=10_000)
    paper.connect()                         # always succeeds
    paper.set_price("EURUSD", bid=1.0850, ask=1.0852)

    result = paper.execute_trade("EURUSD", "BUY", lots=0.10,
                                  stop_loss_pips=20, take_profit_pips=40)
    print(result.ticket, result.price)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from broker.broker_interface import (
    AccountInfo,
    BrokerInterface,
    Position,
    SymbolInfo,
    Tick,
    TradeResult,
)


class PaperBroker(BrokerInterface):
    """
    Fully in-memory paper broker. No external connections required.
    Thread-safe for concurrent scanner + ARIA usage.
    """

    def __init__(
        self,
        initial_balance: float             = 10_000.0,
        leverage:        int               = 100,
        price_feed:      Optional[Callable[..., Tick]] = None,
        notify_callback: Callable          = print,
        spread_pips:     float             = 1.5,
    ):
        self._balance      = initial_balance
        self._leverage     = leverage
        self._price_feed   = price_feed      # callable(symbol) -> Tick | None
        self._notify       = notify_callback
        self._default_spread = spread_pips
        self._connected    = False
        self._lock         = threading.Lock()

        self._prices:    dict[str, Tick]     = {}
        self._positions: dict[int, Position] = {}
        self._closed:    list[Position]      = []
        self._next_ticket = 1000001

    # ── BrokerInterface ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def platform_name(self) -> str:
        return "Paper"

    def connect(self, **_) -> bool:
        self._connected = True
        self._notify("[Paper] Paper broker connected.")
        return True

    def disconnect(self) -> None:
        self._connected = False

    # ── Price management ───────────────────────────────────────────────────

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """Manually set a price for a symbol (useful in tests)."""
        with self._lock:
            self._prices[symbol] = Tick(symbol=symbol, bid=bid, ask=ask, time=datetime.now())

    def get_tick(self, symbol: str) -> Optional[Tick]:
        if self._price_feed:
            tick = self._price_feed(symbol)
            if tick:
                return tick
        with self._lock:
            return self._prices.get(symbol)

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        return None   # paper broker has no historical data by default

    def get_historical_rates(self, symbol: str, timeframe: str = "H1", count: int = 50):
        return None

    def ohclv_data(self, symbol: str, timeframe: str = "H1", num_bars: int = 1000):
        return None

    # ── Account ────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[AccountInfo]:
        with self._lock:
            floating = sum(p.profit for p in self._positions.values())
            equity   = self._balance + floating
            return AccountInfo(
                balance      = round(self._balance, 2),
                equity       = round(equity, 2),
                profit       = round(floating, 2),
                margin       = 0.0,
                margin_free  = round(equity, 2),
                margin_level = 9999.0,
                leverage     = self._leverage,
            )

    # ── Positions ──────────────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        with self._lock:
            self._update_floating()
            return list(self._positions.values())

    def _update_floating(self) -> None:
        """Recalculate unrealised P&L for all open positions (must hold lock)."""
        for pos in self._positions.values():
            tick = self._prices.get(pos.symbol)
            if not tick:
                continue
            if pos.type == 0:   # BUY
                pos.price_current = tick.bid
                pos.profit = round((tick.bid - pos.price_open) * 10000 * pos.volume, 2)
            else:               # SELL
                pos.price_current = tick.ask
                pos.profit = round((pos.price_open - tick.ask) * 10000 * pos.volume, 2)

    # ── Symbol info ────────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        return SymbolInfo(
            name            = symbol,
            spread_pips     = self._default_spread,
            volume_min      = 0.01,
            volume_max      = 100.0,
            volume_step     = 0.01,
            trade_tick_value= 1.0,
            trade_tick_size = 0.00001,
        )

    # ── Order execution ────────────────────────────────────────────────────

    def execute_trade(
        self,
        symbol:           str,
        action:           str,
        lots:             float,
        stop_loss_pips:   float = 0.0,
        take_profit_pips: float = 0.0,
        strategy:         str   = "Unknown",
        magic:            int   = 0,
        comment:          str   = "",
    ) -> TradeResult:
        tick = self.get_tick(symbol)
        if not tick:
            return TradeResult(False, reason=f"No price set for {symbol}")

        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            action_upper = action.upper()
            price    = tick.ask if action_upper == "BUY" else tick.bid
            pip_val  = 0.0001

            sl_price = (price - stop_loss_pips * pip_val)   if (action_upper == "BUY"  and stop_loss_pips)   else \
                       (price + stop_loss_pips * pip_val)   if (action_upper == "SELL" and stop_loss_pips)   else 0.0
            tp_price = (price + take_profit_pips * pip_val) if (action_upper == "BUY"  and take_profit_pips) else \
                       (price - take_profit_pips * pip_val) if (action_upper == "SELL" and take_profit_pips) else 0.0

            self._positions[ticket] = Position(
                ticket        = ticket,
                symbol        = symbol,
                type          = 0 if action_upper == "BUY" else 1,
                volume        = lots,
                price_open    = price,
                price_current = price,
                sl            = round(sl_price, 5),
                tp            = round(tp_price, 5),
                profit        = 0.0,
                strategy      = strategy,
                time          = datetime.now(),
            )

        self.log_trade_history(action_upper, symbol, lots, price, ticket, comment, strategy)
        self._notify(f"[Paper] {action_upper} {lots}L {symbol} @ {price} (#{ticket})")
        return TradeResult(True, ticket=ticket, price=price, volume=lots)

    def close_position(self, symbol: str) -> bool:
        with self._lock:
            to_close = [p for p in self._positions.values() if p.symbol == symbol]
            if not to_close:
                return False
            tick = self._prices.get(symbol)
            for pos in to_close:
                if tick:
                    if pos.type == 0:
                        pos.profit = round((tick.bid - pos.price_open) * 10000 * pos.volume, 2)
                    else:
                        pos.profit = round((pos.price_open - tick.ask) * 10000 * pos.volume, 2)
                self._balance += pos.profit
                self._closed.append(pos)
                del self._positions[pos.ticket]
                self.log_trade_history(
                    "CLOSE", symbol, pos.volume,
                    tick.bid if pos.type == 0 else tick.ask if tick else pos.price_open,
                    pos.ticket, f"Profit: {pos.profit}", pos.strategy, pos.profit,
                )
                self._notify(f"[Paper] Closed {symbol} #{pos.ticket} P&L={pos.profit:+.2f}")
        return True

    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: Optional[float] = None) -> bool:
        with self._lock:
            pos = self._positions.get(ticket)
            if pos is None:
                return False
            pos.sl = new_sl
            if new_tp is not None:
                pos.tp = new_tp
        return True

    def get_daily_realized_profit(self) -> float:
        today = datetime.now().date()
        return sum(
            p.profit for p in self._closed
            if p.time and p.time.date() == today
        )

    # ── Balance management (test helpers) ─────────────────────────────────

    def deposit(self, amount: float) -> None:
        with self._lock:
            self._balance += amount

    def reset(self, balance: float = 10_000.0) -> None:
        with self._lock:
            self._balance   = balance
            self._positions = {}
            self._closed    = []

    # ── Legacy shims ──────────────────────────────────────────────────────

    def getPositions(self):         return self.get_positions()
    def getAccountInfo(self):       return self.get_account_info()
    def get_tick_data(self, sym):   t = self.get_tick(sym); return t.__dict__ if t else None
    def get_total_floating_profit(self): return sum(p.profit for p in self.get_positions())
    def close_all_positions(self):
        for sym in {p.symbol for p in self.get_positions()}:
            self.close_position(sym)
    def _strategy_for(self, ticket): return "Paper"
    def _log_trade_history(self, **kwargs): self.log_trade_history(**kwargs)
    def _mark_cooldown(self, symbol): pass
    def register_position_monitor(self, monitor): pass
    def ohclv_data(self, symbol, timeframe="H1", num_bars=1000): return None
    def get_historical_rates(self, symbol, timeframe="H1", count=50): return None
    