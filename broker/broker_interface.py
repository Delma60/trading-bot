"""
broker/broker_interface.py — Abstract Broker Contract

Every platform adapter (MT5, Alpaca, Interactive Brokers, paper trading…)
must subclass BrokerInterface and implement all abstract methods.

No business logic lives here — only the shape of the API.
BrokerManager is what the rest of the codebase holds onto.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import pandas as pd


# ── Shared data models ────────────────────────────────────────────────────────
# These replace MT5-specific namedtuples throughout the codebase.
# Every adapter maps its native types to these.

@dataclass
class AccountInfo:
    balance:       float
    equity:        float
    profit:        float
    margin:        float
    margin_free:   float
    margin_level:  float   # percent; 0.0 if no open positions
    currency:      str     = "USD"
    leverage:      int     = 100
    server:        str     = ""
    login:         int     = 0


@dataclass
class Position:
    ticket:       int
    symbol:       str
    type:         int        # 0 = BUY, 1 = SELL  (mirrors MT5 convention)
    volume:       float
    price_open:   float
    price_current: float
    sl:           float      # 0.0 = not set
    tp:           float      # 0.0 = not set
    profit:       float
    strategy:     str        = "Unknown"
    magic:        int        = 0
    comment:      str        = ""
    time:         Optional[datetime] = None


@dataclass
class TradeResult:
    success:   bool
    ticket:    int     = 0
    price:     float   = 0.0
    volume:    float   = 0.0
    reason:    str     = ""
    retcode:   int     = 0    # platform-specific return code (0 = success)


@dataclass
class SymbolInfo:
    name:           str
    description:    str    = ""
    category:       str    = "forex"
    digits:         int    = 5
    point:          float  = 0.00001
    spread_pips:    Optional[float] = None
    volume_min:     float  = 0.01
    volume_max:     float  = 100.0
    volume_step:    float  = 0.01
    trade_tick_value: float = 1.0
    trade_tick_size:  float = 0.00001
    filling_mode:   int    = 1       # 1 = FOK, 2 = IOC, 4 = RETURN (bitmask)
    stops_level:    int    = 0       # minimum stop distance in points
    extra:          dict   = field(default_factory=dict)


@dataclass
class Tick:
    symbol: str
    bid:    float
    ask:    float
    last:   float   = 0.0
    volume: float   = 0.0
    time:   Optional[datetime] = None

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ── Abstract interface ────────────────────────────────────────────────────────

class BrokerInterface(ABC):
    """
    Platform-agnostic broker contract.

    Subclass this for every platform. Only implement what the platform
    actually supports — raise NotImplementedError for unsupported features
    rather than returning silent no-ops.

    Thread-safety: implementations must be thread-safe. All methods may be
    called concurrently from scanner threads, ARIA chat, and background daemons.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def connected(self) -> bool:
        """True when the broker connection is live and authenticated."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Human-readable platform identifier, e.g. 'MT5', 'Alpaca', 'Paper'."""

    @abstractmethod
    def connect(self, **credentials) -> bool:
        """
        Authenticate and establish a session.
        Returns True on success. Implementations should store credentials
        internally for reconnect() calls.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly close the connection. Safe to call even when already disconnected."""

    def reconnect(self) -> bool:
        """
        Attempt to re-establish a lost connection using stored credentials.
        Default: calls disconnect() then connect() with no arguments.
        Override if the platform has a dedicated reconnect mechanism.
        """
        self.disconnect()
        return self.connect()

    def ensure_connected(self) -> bool:
        """
        Return True if connected, otherwise attempt a reconnect.
        Override for platforms that support ping-style health checks.
        """
        if self.connected:
            return True
        return self.reconnect()

    # ── Account ───────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """Current account snapshot. Returns None if unavailable."""

    # ── Positions ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """All currently open positions. Returns empty list (never None)."""

    def get_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Convenience: filter open positions by symbol."""
        return [p for p in self.get_positions() if p.symbol == symbol]

    # ── Market data ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Latest bid/ask snapshot. Returns None if the symbol is not quoted."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str,       # "M1","M5","M15","M30","H1","H4","D1","W1","MN"
        count:     int = 500,
    ) -> Optional["pd.DataFrame"]:
        """
        Historical OHLCV bars as a DataFrame with columns:
        open, high, low, close, volume (DatetimeIndex).
        Returns None on failure.
        """

    # ── Symbol catalogue ──────────────────────────────────────────────────────

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Static metadata for one symbol. Returns None if not found."""

    def search_symbols(
        self,
        query:       Optional[str]  = None,
        category:    Optional[str]  = None,
        max_results: int            = 50,
    ) -> list[SymbolInfo]:
        """
        Search the broker's symbol catalogue.
        Default implementation returns an empty list.
        Override to provide filtered results.
        """
        return []

    # ── Order execution ───────────────────────────────────────────────────────

    @abstractmethod
    def execute_trade(
        self,
        symbol:           str,
        action:           str,    # "BUY" or "SELL"
        lots:             float,
        stop_loss_pips:   float = 0.0,
        take_profit_pips: float = 0.0,
        strategy:         str   = "Unknown",
        magic:            int   = 0,
        comment:          str   = "",
    ) -> TradeResult:
        """Place a market order. Returns a TradeResult (never raises)."""

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Close all positions for a symbol. Returns True if at least one was closed."""

    def close_all_positions(self) -> list[TradeResult]:
        """
        Close every open position.
        Default: iterates get_positions() and calls close_position() per symbol.
        Override for platforms with a single-call flatten endpoint.
        """
        symbols = list({p.symbol for p in self.get_positions()})
        results = []
        for sym in symbols:
            ok = self.close_position(sym)
            results.append(TradeResult(success=ok, reason="closed" if ok else "failed"))
        return results

    def close_profitable_positions(
        self, symbol: Optional[str] = None
    ) -> list[TradeResult]:
        """
        Close positions that are currently in profit.
        Default: filters get_positions() and calls close_position() per symbol.
        """
        positions = self.get_positions()
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        profitable = [p for p in positions if p.profit > 0]
        results = []
        for pos in profitable:
            ok = self.close_position(pos.symbol)
            results.append(TradeResult(success=ok))
        return results

    def modify_position(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: Optional[float] = None,
    ) -> bool:
        """
        Modify the SL (and optionally TP) of an open position.
        Returns True on success. Default: not supported.
        """
        raise NotImplementedError(f"{self.platform_name} does not support modify_position()")

    def partial_close_position(
        self,
        ticket:      int,
        symbol:      str,
        close_ratio: float = 0.5,
    ) -> TradeResult:
        """
        Close a fraction of a position's volume.
        Default: not supported.
        """
        raise NotImplementedError(f"{self.platform_name} does not support partial_close()")

    # ── Cooldown ──────────────────────────────────────────────────────────────

    def set_cooldown(self, seconds: int) -> None:
        """Set the post-close cooldown period. Optional — no-op by default."""

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        """Returns (in_cooldown, remaining_seconds). No-op default: (False, 0)."""
        return False, 0.0

    # ── Position monitor hooks ────────────────────────────────────────────────

    def register_position_monitor(self, monitor: Any) -> None:
        """
        Wire up a PositionMonitor so bot-closes are excluded from
        external-close callbacks. Optional — no-op by default.
        """

    # ── Trade history ─────────────────────────────────────────────────────────

    def get_daily_realized_profit(self) -> float:
        """Today's total realized P&L. Returns 0.0 if unavailable."""
        return 0.0

    def get_total_floating_profit(self) -> float:
        """Sum of profit across all open positions."""
        return sum(p.profit for p in self.get_positions())

    # ── Misc helpers (used by RiskManager, ProfitGuard, etc.) ─────────────────

    def get_history_deals(self, start_ts: int, end_ts: int) -> list[dict]:
        """
        Closed deal history between two Unix timestamps.
        Returns list of dicts with at least: profit, position_id, entry, price.
        Default: empty list.
        """
        return []

    def strategy_for_ticket(self, ticket: int) -> str:
        """Look up which strategy opened a ticket. Default: 'Unknown'."""
        return "Unknown"

    # ── Internal helpers (for adapters that need it) ──────────────────────────

    def log_trade_history(
        self,
        action:   str,
        symbol:   str,
        lots:     float,
        price:    float,
        ticket:   int,
        comment:  str,
        strategy: str   = "Unknown",
        profit:   Optional[float] = None,
    ) -> None:
        """
        Persist a trade record to CSV / DB.
        Default: writes to data/trade_history.csv (same format as existing code).
        Override to write to a different store.
        """
        import csv
        from pathlib import Path

        path = Path("data/trade_history.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()

        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "Timestamp", "Ticket", "Action", "Symbol",
                    "Volume", "Execution_Price", "Comment", "Strategy", "Profit",
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticket, action, symbol, lots, price, comment, strategy,
                profit if profit is not None else "",
            ])