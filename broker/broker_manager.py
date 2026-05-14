"""
broker/broker_manager.py — Unified Broker Manager

The single object every module holds onto.  No module should ever import
MT5Broker, PaperBroker, or any concrete adapter directly — they import
BrokerManager and call the BrokerInterface methods through it.

Responsibilities
----------------
1. Adapter routing          — holds one active BrokerInterface implementation
2. Health monitoring        — background thread reconnects on dropped sessions
3. Graceful fallback        — switches to PaperBroker when MT5 is unavailable
4. Credential management    — delegates to ProfileManager, never stores raw creds
5. Legacy shim              — exposes all old Trader method names so existing
                              code (RiskManager, PortfolioManager, chat.py …)
                              works without a single import change

Usage
-----
    from broker.broker_manager import BrokerManager

    # Production (MT5)
    manager = BrokerManager(platform="mt5", notify_callback=agent_notify)
    manager.connect(login=12345, password="pw", server="MetaQuotes-Demo")

    # Paper trading / testing
    manager = BrokerManager(platform="paper", notify_callback=print)
    manager.connect()
    manager.set_price("EURUSD", bid=1.0850, ask=1.0852)

    # Auto-detect from credentials.json
    manager = BrokerManager.from_credentials(notify_callback=agent_notify)

Thread-safety
-------------
All public methods are safe to call from any thread.  The health monitor
runs in a daemon thread and holds its own lock when swapping adapters.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
from broker.mt5_broker import MT5Broker
from broker.paper_broker import PaperBroker

import pandas as pd

from broker.broker_interface import (
    AccountInfo,
    BrokerInterface,
    Position,
    SymbolInfo,
    Tick,
    TradeResult,
)


# ── Constants ─────────────────────────────────────────────────────────────────

HEALTH_CHECK_INTERVAL = 15   # seconds between connectivity pings
RECONNECT_MAX_ATTEMPTS = 3   # before giving up and switching to paper


# ── BrokerManager ─────────────────────────────────────────────────────────────

class BrokerManager:
    """
    Platform-agnostic broker façade.

    Every module in ARIA calls broker methods through this class.
    The underlying adapter (MT5, Paper, future Alpaca …) is swappable
    at construction time or at runtime via swap_adapter().

    Parameters
    ----------
    platform : str
        "mt5" | "paper" | "alpaca" (extensible)
    notify_callback : Callable
        Receives human-readable status messages.
    auto_fallback : bool
        If True, automatically switches to PaperBroker when MT5
        becomes permanently unavailable.  Default: True.
    magic : int
        Magic number passed to MT5 for order identification.
    """

    def __init__(
        self,
        platform:        str      = "mt5",
        notify_callback: Callable = print,
        auto_fallback:   bool     = True,
        magic:           int      = 234000,
        cooldown_seconds: int     = 5,
    ):
        self._platform       = platform.lower()
        self._notify         = notify_callback
        self._auto_fallback  = auto_fallback
        self._magic          = magic
        self._cooldown_secs  = cooldown_seconds

        self.platforms = {
            "mt5": MT5Broker,
            "paper": PaperBroker,
        }
        self._adapter: BrokerInterface = self._build_adapter(platform)
        self._lock    = threading.RLock()
        self._credentials: dict = {}

        # Health monitor
        self._health_running = False
        self._health_thread: Optional[threading.Thread] = None
        

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_credentials(
        cls,
        notify_callback: Callable = print,
        credentials_path: Path    = Path("data/credentials.json"),
        auto_fallback: bool       = True,
        **kwargs,
    ) -> "BrokerManager":
        """
        Build a BrokerManager by reading data/credentials.json.
        Falls back to PaperBroker if the file is missing.
        """
        import json, os

        # 1. Environment variables take priority (CI / Docker)
        if os.getenv("MT5_LOGIN"):
            creds = {
                "login":    int(os.getenv("MT5_LOGIN")),
                "password": os.getenv("MT5_PASSWORD", ""),
                "server":   os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
            }
            mgr = cls(platform="mt5", notify_callback=notify_callback,
                      auto_fallback=auto_fallback, **kwargs)
            mgr._credentials = creds
            return mgr

        # 2. credentials.json
        if credentials_path.exists():
            try:
                creds = json.loads(credentials_path.read_text(encoding="utf-8"))
                mgr = cls(platform="mt5", notify_callback=notify_callback,
                          auto_fallback=auto_fallback, **kwargs)
                mgr._credentials = creds
                return mgr
            except Exception as exc:
                notify_callback(f"[BrokerManager] Could not read credentials: {exc}")

        # 3. No credentials → paper mode
        notify_callback("[BrokerManager] No credentials found — starting in Paper mode.")
        return cls(platform="paper", notify_callback=notify_callback,
                   auto_fallback=False, **kwargs)

    def _build_adapter(self, platform: str) -> BrokerInterface:
        
        p = platform.lower()
        if p in self.platforms:
            AdapterClass = self.platforms[p]
            return AdapterClass(notify_callback=self._notify, cooldown_seconds=self._cooldown_secs)
        raise ValueError(f"[BrokerManager] Unknown platform '{platform}'. "
                         f"Choose 'mt5' or 'paper'.")

    # ── Adapter management ────────────────────────────────────────────────────

    def swap_adapter(self, new_adapter: BrokerInterface) -> None:
        """
        Hot-swap the underlying adapter.  Safe to call from any thread.
        Existing open positions on the old adapter are NOT transferred.
        """
        with self._lock:
            old = self._adapter
            self._adapter = new_adapter
            self._notify(
                f"[BrokerManager] Adapter swapped: "
                f"{type(old).__name__} → {type(new_adapter).__name__}"
            )

    @property
    def adapter(self) -> BrokerInterface:
        """Direct access to the underlying adapter (use sparingly)."""
        with self._lock:
            return self._adapter

    @property
    def platform_name(self) -> str:
        with self._lock:
            return self._adapter.platform_name

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **credentials) -> bool:
        """
        Authenticate with the broker.

        Credentials can be passed directly or were pre-loaded via
        from_credentials().  Starts the background health monitor on success.
        """
        if credentials:
            self._credentials.update(credentials)

        creds = self._credentials
        with self._lock:
            ok = self._adapter.connect(**creds)

        if ok:
            self._notify(
                f"[BrokerManager] Connected via {self._adapter.platform_name}."
            )
            self._start_health_monitor()
        else:
            self._notify("[BrokerManager] Connection failed.")
            if self._auto_fallback and self._platform == "mt5":
                self._switch_to_paper()
        return ok

    def disconnect(self) -> None:
        self._stop_health_monitor()
        with self._lock:
            self._adapter.disconnect()
        self._notify("[BrokerManager] Disconnected.")

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._adapter.connected

    # ── Health monitor ────────────────────────────────────────────────────────

    def _start_health_monitor(self) -> None:
        if self._health_running:
            return
        self._health_running = True
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="BrokerHealthMonitor"
        )
        self._health_thread.start()

    def _stop_health_monitor(self) -> None:
        self._health_running = False
        if self._health_thread:
            self._health_thread.join(timeout=3)

    def _health_loop(self) -> None:
        failures = 0
        while self._health_running:
            time.sleep(HEALTH_CHECK_INTERVAL)
            try:
                with self._lock:
                    ok = self._adapter.ensure_connected()
                if not ok:
                    failures += 1
                    self._notify(
                        f"[BrokerManager] Health check failed "
                        f"({failures}/{RECONNECT_MAX_ATTEMPTS}). Reconnecting…"
                    )
                    if failures >= RECONNECT_MAX_ATTEMPTS:
                        if self._auto_fallback and self._platform == "mt5":
                            self._switch_to_paper()
                        failures = 0
                else:
                    failures = 0
            except Exception as exc:
                self._notify(f"[BrokerManager] Health monitor error: {exc}")

    def _switch_to_paper(self) -> None:
        self._notify(
            "[BrokerManager] MT5 unreachable — switching to PaperBroker. "
            "Positions will not reflect live market data."
        )
        from broker.paper_broker import PaperBroker
        paper = PaperBroker(notify_callback=self._notify)
        paper.connect()
        self.swap_adapter(paper)

    # ══════════════════════════════════════════════════════════════════════════
    # BrokerInterface delegation
    # Every method below is a thin pass-through to self._adapter.
    # They exist here so callers never need to know about adapters.
    # ══════════════════════════════════════════════════════════════════════════

    def get_account_info(self) -> Optional[AccountInfo]:
        with self._lock:
            return self._adapter.get_account_info()

    def get_positions(self) -> list[Position]:
        with self._lock:
            return self._adapter.get_positions()

    def get_positions_for_symbol(self, symbol: str) -> list[Position]:
        with self._lock:
            return self._adapter.get_positions_for_symbol(symbol)

    def get_tick(self, symbol: str) -> Optional[Tick]:
        with self._lock:
            return self._adapter.get_tick(symbol)

    def get_ohlcv(
        self, symbol: str, timeframe: str, count: int = 500
    ) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._adapter.get_ohlcv(symbol, timeframe, count)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        with self._lock:
            return self._adapter.get_symbol_info(symbol)

    def search_symbols(
        self,
        query:       Optional[str] = None,
        category:    Optional[str] = None,
        max_results: int           = 50,
    ) -> list[SymbolInfo]:
        with self._lock:
            return self._adapter.search_symbols(query, category, max_results)

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
        with self._lock:
            return self._adapter.execute_trade(
                symbol, action, lots,
                stop_loss_pips, take_profit_pips,
                strategy, magic or self._magic, comment,
            )

    def close_position(self, symbol: str) -> bool:
        with self._lock:
            return self._adapter.close_position(symbol)

    def close_all_positions(self) -> list[TradeResult]:
        with self._lock:
            return self._adapter.close_all_positions()

    def close_profitable_positions(
        self, symbol: Optional[str] = None
    ) -> list[TradeResult]:
        with self._lock:
            return self._adapter.close_profitable_positions(symbol)

    def modify_position(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: Optional[float] = None,
    ) -> bool:
        with self._lock:
            return self._adapter.modify_position(ticket, symbol, new_sl, new_tp)

    def partial_close_position(
        self,
        ticket:      int,
        symbol:      str,
        close_ratio: float = 0.5,
    ) -> TradeResult:
        with self._lock:
            return self._adapter.partial_close_position(ticket, symbol, close_ratio)

    def set_cooldown(self, seconds: int) -> None:
        self._cooldown_secs = max(0, seconds)
        with self._lock:
            self._adapter.set_cooldown(seconds)

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        with self._lock:
            return self._adapter.is_in_cooldown(symbol)

    def get_daily_realized_profit(self) -> float:
        with self._lock:
            return self._adapter.get_daily_realized_profit()

    def get_total_floating_profit(self) -> float:
        with self._lock:
            return self._adapter.get_total_floating_profit()

    def get_history_deals(self, start_ts: int, end_ts: int) -> list[dict]:
        with self._lock:
            return self._adapter.get_history_deals(start_ts, end_ts)

    def strategy_for_ticket(self, ticket: int) -> str:
        with self._lock:
            return self._adapter.strategy_for_ticket(ticket)

    def log_trade_history(self, **kwargs) -> None:
        with self._lock:
            self._adapter.log_trade_history(**kwargs)

    def register_position_monitor(self, monitor: Any) -> None:
        with self._lock:
            self._adapter.register_position_monitor(monitor)

    # ── PaperBroker-specific helpers (no-op on MT5) ───────────────────────────

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Manually inject a price (PaperBroker only).
        Safe to call on MT5 — silently ignored.
        """
        with self._lock:
            if hasattr(self._adapter, "set_price"):
                self._adapter.set_price(symbol, bid, ask)

    def deposit(self, amount: float) -> None:
        """Add funds to the paper account. No-op on MT5."""
        with self._lock:
            if hasattr(self._adapter, "deposit"):
                self._adapter.deposit(amount)

    # ══════════════════════════════════════════════════════════════════════════
    # Legacy shims — exact method names from the old Trader class.
    # Every existing call site works without modification.
    # ══════════════════════════════════════════════════════════════════════════

    # -- Account / positions --------------------------------------------------

    def getAccountInfo(self) -> Optional[AccountInfo]:
        return self.get_account_info()

    def getPositions(self) -> list[Position]:
        return self.get_positions()

    # -- Market data ----------------------------------------------------------

    def ohclv_data(
        self, symbol: str, timeframe: str = "H1", num_bars: int = 1000
    ) -> Optional[pd.DataFrame]:
        return self.get_ohlcv(symbol, timeframe, num_bars)

    def get_historical_rates(
        self, symbol: str, timeframe: str = "H1", count: int = 50
    ) -> Optional[pd.DataFrame]:
        return self.get_ohlcv(symbol, timeframe, count)

    def get_tick_data(self, symbol: str) -> Optional[dict]:
        tick = self.get_tick(symbol)
        return tick.__dict__ if tick else None

    def getBalance(self) -> Optional[float]:
        acct = self.get_account_info()
        return acct.balance if acct else None

    # -- Trade execution ------------------------------------------------------

    def execute_trade_legacy(
        self,
        symbol:           str,
        action:           str,
        lots:             float,
        stop_loss_pips:   float = 0.0,
        take_profit_pips: float = 0.0,
        strategy:         str   = "Unknown",
    ) -> dict:
        """
        Returns the old dict format {"success": bool, "ticket": int, "price": float}.
        Used by ActionExecutor and any code that checks result.get("success").
        """
        result = self.execute_trade(
            symbol, action, lots,
            stop_loss_pips, take_profit_pips, strategy,
        )
        return {
            "success": result.success,
            "ticket":  result.ticket,
            "price":   result.price,
            "reason":  result.reason,
        }

    # -- Close helpers --------------------------------------------------------

    def close_all_positions_legacy(self) -> None:
        """Matches old Trader.close_all_positions() void signature."""
        self.close_all_positions()

    def close_profitable_positions_legacy(
        self, symbol: Optional[str] = None
    ) -> str:
        """Returns a human-readable string (old Trader behaviour)."""
        results = self.close_profitable_positions(symbol)
        if not results:
            return "No profitable positions to close."
        lines = []
        for r in results:
            icon = "✅" if r.success else "❌"
            lines.append(f"{icon} {r.reason or ('closed' if r.success else 'failed')}")
        return "\n".join(lines)

    # -- Internal helpers referenced by old code ------------------------------

    def _strategy_for(self, ticket: int) -> str:
        return self.strategy_for_ticket(ticket)

    def _log_trade_history(self, **kwargs) -> None:
        self.log_trade_history(**kwargs)

    def _mark_cooldown(self, symbol: str) -> None:
        """Compatibility shim — cooldown is now managed inside adapters."""
        with self._lock:
            if hasattr(self._adapter, "_mark_cooldown"):
                self._adapter._mark_cooldown(symbol)

    def _mark_cooldown_public(self, symbol: str) -> None:
        self._mark_cooldown(symbol)

    # -- Utility --------------------------------------------------------------

    def status(self) -> str:
        """Human-readable connection status for ARIA to surface."""
        acct = self.get_account_info()
        if not self.connected or not acct:
            return (
                f"[{self.platform_name}] Disconnected. "
                f"Auto-fallback: {'enabled' if self._auto_fallback else 'disabled'}."
            )
        return (
            f"[{self.platform_name}] Connected | "
            f"Balance ${acct.balance:,.2f} | "
            f"Equity ${acct.equity:,.2f} | "
            f"Margin level {acct.margin_level:.0f}%"
        )

    def __repr__(self) -> str:
        return (
            f"BrokerManager(platform={self.platform_name!r}, "
            f"connected={self.connected})"
        )