"""
broker/mt5_broker.py — MetaTrader 5 Adapter

Wraps the existing Trader class and maps everything to the
BrokerInterface types. Zero behaviour changes — just a thin
translation layer so the rest of the codebase can stay generic.

Usage
-----
    from broker.mt5_broker import MT5Broker
    broker = MT5Broker(notify_callback=agent_notify, magic=234000)
    broker.connect(login=12345, password="pw", server="MetaQuotes-Demo")
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from broker.broker_interface import (
    AccountInfo,
    BrokerInterface,
    Position,
    SymbolInfo,
    Tick,
    TradeResult,
)


class MT5Broker(BrokerInterface):
    """
    MetaTrader 5 adapter.

    Wraps the raw MT5 Python library (not the old Trader class) so we
    have one place to contain all MT5-specific quirks:
      - Filling-mode detection (FOK / IOC / RETURN bitmask)
      - Pip-multiplier inference
      - Cooldown tracking
      - Ticket → strategy mapping
    """

    def __init__(
        self,
        notify_callback: Callable = print,
        magic:           int      = 234000,
        cooldown_seconds: int     = 5,
    ):
        self._notify           = notify_callback
        self._magic            = magic
        self._connected        = False
        self._credentials: dict = {}
        self._cooldown_secs    = cooldown_seconds

        self._execution_lock   = threading.Lock()
        self._pending_orders:  set[str]           = set()
        self._cooldown_until:  dict[str, datetime] = {}
        self._cooldown_lock    = threading.Lock()
        self._ticket_strategy: dict[int, str]      = {}
        self._ticket_strategy_path = Path("data/cache/ticket_strategy.json")

        self._position_monitor: Optional[Any] = None

        self._load_ticket_strategy()

    # ── BrokerInterface properties ─────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def platform_name(self) -> str:
        return "MT5"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, login: int = 0, password: str = "", server: str = "MetaQuotes-Demo", **_) -> bool:
        import os
        import MetaTrader5 as mt5

        self._credentials = {"login": login, "password": password, "server": server}

        mt5_path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        init_ok  = (
            mt5.initialize(path=mt5_path, timeout=60000)
            if os.path.exists(mt5_path)
            else mt5.initialize(timeout=60000)
        )
        if not init_ok:
            self._notify(f"[MT5] Init failed: {mt5.last_error()}")
            return False

        if not mt5.login(login, password=password, server=server):
            self._notify(f"[MT5] Login failed: {mt5.last_error()}")
            return False

        self._connected = True
        self._notify(f"[MT5] Connected — account #{login} on {server}")
        return True

    def disconnect(self) -> None:
        import MetaTrader5 as mt5
        mt5.shutdown()
        self._connected = False

    def ensure_connected(self) -> bool:
        import MetaTrader5 as mt5
        if not self._connected:
            return False
        try:
            if mt5.terminal_info() is None:
                self._notify("[MT5] Connection lost — reconnecting…")
                creds = self._credentials
                return self.connect(
                    login    = creds.get("login",    0),
                    password = creds.get("password", ""),
                    server   = creds.get("server",   "MetaQuotes-Demo"),
                )
        except Exception as exc:
            self._notify(f"[MT5] Health check failed: {exc}")
            return False
        return True

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[AccountInfo]:
        import MetaTrader5 as mt5
        if not self._connected:
            return None
        a = mt5.account_info()
        if a is None:
            return None
        return AccountInfo(
            balance      = float(a.balance),
            equity       = float(a.equity),
            profit       = float(a.profit),
            margin       = float(getattr(a, "margin", 0.0)),
            margin_free  = float(getattr(a, "margin_free", 0.0)),
            margin_level = float(a.margin_level) if a.margin_level else 0.0,
            currency     = str(getattr(a, "currency", "USD")),
            leverage     = int(getattr(a, "leverage", 100)),
            server       = str(getattr(a, "server", "")),
            login        = int(getattr(a, "login", 0)),
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        import MetaTrader5 as mt5
        if not self._connected:
            return []
        raw = mt5.positions_get()
        if not raw:
            return []
        return [
            Position(
                ticket        = p.ticket,
                symbol        = p.symbol,
                type          = p.type,       # 0 = BUY, 1 = SELL
                volume        = float(p.volume),
                price_open    = float(p.price_open),
                price_current = float(p.price_current),
                sl            = float(p.sl or 0.0),
                tp            = float(p.tp or 0.0),
                profit        = float(p.profit),
                strategy      = self.strategy_for_ticket(p.ticket),
                magic         = int(p.magic),
                comment       = str(p.comment),
                time          = datetime.fromtimestamp(p.time) if hasattr(p, "time") else None,
            )
            for p in raw
        ]

    # ── Market data ───────────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[Tick]:
        import MetaTrader5 as mt5
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            return None
        return Tick(
            symbol = symbol,
            bid    = float(t.bid),
            ask    = float(t.ask),
            last   = float(getattr(t, "last", 0.0)),
            volume = float(getattr(t, "volume", 0.0)),
        )

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        import MetaTrader5 as mt5
        if not self._connected:
            return None

        tf_map = {
            "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,  "W1":  mt5.TIMEFRAME_W1,
            "MN":  mt5.TIMEFRAME_MN1,
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates  = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df["time"], unit="s")
        df.drop(columns=["time"], errors="ignore", inplace=True)

        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            elif "real_volume" in df.columns:
                df["volume"] = df["real_volume"]
        return df

    # Alias used by legacy code (get_historical_rates)
    def get_historical_rates(self, symbol: str, timeframe: str = "H1", count: int = 50) -> Optional[pd.DataFrame]:
        return self.get_ohlcv(symbol, timeframe, count)

    # Alias used by LocalCache (ohclv_data)
    def ohclv_data(self, symbol: str, timeframe: str = "H1", num_bars: int = 1000) -> Optional[pd.DataFrame]:
        return self.get_ohlcv(symbol, timeframe, num_bars)

    # ── Symbol catalogue ──────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        import MetaTrader5 as mt5
        raw = mt5.symbol_info(symbol)
        if raw is None:
            return None
        return SymbolInfo(
            name              = symbol,
            description       = str(getattr(raw, "description", "")),
            digits            = int(raw.digits),
            point             = float(raw.point),
            volume_min        = float(raw.volume_min),
            volume_max        = float(raw.volume_max),
            volume_step       = float(raw.volume_step),
            trade_tick_value  = float(raw.trade_tick_value),
            trade_tick_size   = float(raw.trade_tick_size),
            filling_mode      = int(raw.filling_mode),
            stops_level       = int(raw.trade_stops_level),
            spread_pips       = self._live_spread_pips(symbol, raw),
            extra             = raw._asdict() if hasattr(raw, "_asdict") else {},
        )

    def search_symbols(
        self,
        query:       Optional[str] = None,
        category:    Optional[str] = None,
        max_results: int           = 50,
    ) -> list[SymbolInfo]:
        import MetaTrader5 as mt5
        all_syms = mt5.symbols_get() or []
        results  = []
        q = query.upper().strip() if query else None

        for sym in all_syms:
            name = sym.name
            desc = getattr(sym, "description", "") or ""
            cat  = _infer_category(name)
            if category and cat != category.lower():
                continue
            if q and q not in name.upper() and q not in desc.upper():
                continue

            info = self.get_symbol_info(name)
            if info:
                info.category = cat
                info.description = desc
                results.append(info)
            if len(results) >= max_results:
                break
        return results

    # ── Order execution ───────────────────────────────────────────────────────

    def execute_trade(
        self,
        symbol:           str,
        action:           str,
        lots:             float,
        stop_loss_pips:   float = 0.0,
        take_profit_pips: float = 0.0,
        strategy:         str   = "Unknown",
        magic:            int   = 0,
        comment:          str   = "AI Bot Trade",
    ) -> TradeResult:
        import MetaTrader5 as mt5

        if not self.ensure_connected():
            return TradeResult(False, reason="MT5 not connected")

        in_cd, remaining = self.is_in_cooldown(symbol)
        if in_cd:
            return TradeResult(False, reason=f"{symbol} in cooldown — {remaining:.1f}s remaining")

        order_key = f"{symbol}_{action.upper()}"
        with self._execution_lock:
            if order_key in self._pending_orders:
                return TradeResult(False, reason=f"Order for {symbol} already in flight")
            self._pending_orders.add(order_key)

        try:
            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                return TradeResult(False, reason=f"{symbol} not found on broker")
            if not sym_info.visible:
                mt5.symbol_select(symbol, True)

            order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            tick       = mt5.symbol_info_tick(symbol)
            if not tick:
                return TradeResult(False, reason=f"No tick data for {symbol}")

            price        = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            point        = sym_info.point
            pip_mult     = self._pip_multiplier(symbol)
            digits       = sym_info.digits
            min_stop     = sym_info.trade_stops_level or 0
            spread_pts   = int(round((tick.ask - tick.bid) / point))
            safe_dist    = spread_pts + min_stop

            sl_pts = max(int(stop_loss_pips   * pip_mult), safe_dist) if stop_loss_pips   > 0 else 0
            tp_pts = max(int(take_profit_pips * pip_mult), safe_dist) if take_profit_pips > 0 else 0

            if order_type == mt5.ORDER_TYPE_BUY:
                sl_price = round(tick.ask - sl_pts * point, digits) if sl_pts else 0.0
                tp_price = round(tick.ask + tp_pts * point, digits) if tp_pts else 0.0
            else:
                sl_price = round(tick.bid + sl_pts * point, digits) if sl_pts else 0.0
                tp_price = round(tick.bid - tp_pts * point, digits) if tp_pts else 0.0

            fm = sym_info.filling_mode
            if fm & 1:   type_filling = mt5.ORDER_FILLING_FOK
            elif fm & 2: type_filling = mt5.ORDER_FILLING_IOC
            else:        type_filling = mt5.ORDER_FILLING_RETURN

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       float(lots),
                "type":         order_type,
                "price":        price,
                "sl":           sl_price,
                "tp":           tp_price,
                "deviation":    20,
                "magic":        magic or self._magic,
                "comment":      comment,
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }

            with self._execution_lock:
                if not self._connected or mt5.terminal_info() is None:
                    return TradeResult(False, reason="Connection lost during execution")
                result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self._ticket_strategy[result.order] = strategy
                self._save_ticket_strategy()
                self.log_trade_history(
                    action   = action.upper(),
                    symbol   = symbol,
                    lots     = lots,
                    price    = result.price,
                    ticket   = result.order,
                    comment  = comment,
                    strategy = strategy,
                )
                return TradeResult(success=True, ticket=result.order, price=result.price, volume=lots)

            return TradeResult(
                False,
                retcode = result.retcode,
                reason  = f"MT5 {result.retcode}: {result.comment}",
            )

        finally:
            with self._execution_lock:
                self._pending_orders.discard(order_key)

    def close_position(self, symbol: str) -> bool:
        import MetaTrader5 as mt5
        if not self.ensure_connected():
            return False

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False

        success = True
        with self._execution_lock:
            for pos in mt5.positions_get(symbol=symbol) or []:
                sym_info = mt5.symbol_info(symbol)
                tick     = mt5.symbol_info_tick(symbol)
                if not sym_info or not tick:
                    success = False
                    continue

                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price      = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

                fm = sym_info.filling_mode
                if fm & 1:   type_filling = mt5.ORDER_FILLING_FOK
                elif fm & 2: type_filling = mt5.ORDER_FILLING_IOC
                else:        type_filling = mt5.ORDER_FILLING_RETURN

                result = mt5.order_send({
                    "action":       mt5.TRADE_ACTION_DEAL,
                    "symbol":       symbol,
                    "volume":       pos.volume,
                    "type":         order_type,
                    "position":     pos.ticket,
                    "price":        price,
                    "deviation":    20,
                    "magic":        self._magic,
                    "comment":      "Bot close",
                    "type_time":    mt5.ORDER_TIME_GTC,
                    "type_filling": type_filling,
                })

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.log_trade_history(
                        "CLOSE", symbol, pos.volume, result.price,
                        pos.ticket, f"Profit: {pos.profit}",
                        strategy=self.strategy_for_ticket(pos.ticket),
                        profit=pos.profit,
                    )
                    self._mark_cooldown(symbol)
                    if self._position_monitor:
                        self._position_monitor.mark_bot_closed(pos.ticket)
                else:
                    success = False

        return success

    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: Optional[float] = None) -> bool:
        import MetaTrader5 as mt5
        if not self.ensure_connected():
            return False

        with self._execution_lock:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            pos = positions[0]
            tp_value = float(new_tp) if new_tp is not None else float(pos.tp or 0.0)
            result   = mt5.order_send({
                "action":   mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol":   symbol,
                "sl":       float(new_sl),
                "tp":       tp_value,
                "magic":    self._magic,
            })

        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def partial_close_position(self, ticket: int, symbol: str, close_ratio: float = 0.5) -> TradeResult:
        import MetaTrader5 as mt5
        if not (0.1 <= close_ratio <= 0.9):
            return TradeResult(False, reason="close_ratio must be between 0.1 and 0.9")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return TradeResult(False, reason=f"Ticket #{ticket} not found")

        pos     = positions[0]
        vol     = round(pos.volume * close_ratio, 2)
        si      = mt5.symbol_info(symbol)
        if si and vol < si.volume_min:
            return TradeResult(False, reason="Volume below broker minimum")

        tick  = mt5.symbol_info_tick(symbol)
        price = tick.bid if pos.type == 0 else tick.ask
        result = mt5.order_send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       vol,
            "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        self._magic,
            "comment":      "Partial close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_trade_history("PARTIAL_CLOSE", symbol, vol, result.price, result.order, "", profit=None)
            return TradeResult(True, ticket=result.order, price=result.price, volume=vol)
        return TradeResult(False, reason=result.comment)

    # ── Cooldown ──────────────────────────────────────────────────────────────

    def set_cooldown(self, seconds: int) -> None:
        self._cooldown_secs = max(0, seconds)

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        with self._cooldown_lock:
            expiry = self._cooldown_until.get(symbol)
            if expiry is None:
                return False, 0.0
            remaining = (expiry - datetime.now()).total_seconds()
            if remaining <= 0:
                del self._cooldown_until[symbol]
                return False, 0.0
            return True, remaining

    def _mark_cooldown(self, symbol: str) -> None:
        if self._cooldown_secs <= 0:
            return
        with self._cooldown_lock:
            self._cooldown_until[symbol] = datetime.now() + timedelta(seconds=self._cooldown_secs)

    # ── Position monitor ──────────────────────────────────────────────────────

    def register_position_monitor(self, monitor: Any) -> None:
        self._position_monitor = monitor

    # ── Trade history ─────────────────────────────────────────────────────────

    def get_daily_realized_profit(self) -> float:
        import MetaTrader5 as mt5
        if not self._connected:
            return 0.0
        now   = datetime.now()
        start = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        end   = int(now.timestamp())
        deals = mt5.history_deals_get(start, end)
        return sum(d.profit for d in (deals or []) if d.profit is not None)

    def get_history_deals(self, start_ts: int, end_ts: int) -> list[dict]:
        import MetaTrader5 as mt5
        deals = mt5.history_deals_get(start_ts, end_ts)
        if not deals:
            return []
        return [
            {
                "profit":      float(d.profit),
                "position_id": int(d.position_id),
                "ticket":      int(d.ticket),
                "entry":       int(d.entry),
                "price":       float(d.price),
                "commission":  float(getattr(d, "commission", 0.0)),
                "fee":         float(getattr(d, "fee", 0.0)),
                "symbol":      str(d.symbol),
            }
            for d in deals
        ]

    def strategy_for_ticket(self, ticket: int) -> str:
        strat = self._ticket_strategy.get(ticket)
        if strat and strat != "Unknown":
            return strat

        import csv
        from pathlib import Path
        path = Path("data/trade_history.csv")
        if path.exists():
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for row in csv.DictReader(f):
                        if row.get("Ticket") == str(ticket):
                            found = row.get("Strategy", "Unknown")
                            if found != "Unknown":
                                self._ticket_strategy[ticket] = found
                            return found
            except Exception:
                pass
        return "Unknown"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _live_spread_pips(self, symbol: str, sym_info) -> Optional[float]:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if not tick or sym_info.point == 0:
            return None
        spread_pts = (tick.ask - tick.bid) / sym_info.point
        divisor    = 10.0 if sym_info.digits in (5, 3) else 1.0
        return round(spread_pts / divisor, 1)

    @staticmethod
    def _pip_multiplier(symbol: str) -> float:
        s = symbol.upper()
        if any(s.startswith(p) for p in ["XAU", "XAG", "XPT", "XPD"]):
            return 1.0
        if any(t in s for t in ["BTC", "ETH", "LTC", "XBT", "USDT", "DOGE"]):
            return 1.0
        if "30" in s or "500" in s:
            return 1.0
        return 10.0

    def _load_ticket_strategy(self) -> None:
        import json
        if self._ticket_strategy_path.exists():
            try:
                with open(self._ticket_strategy_path, encoding="utf-8") as f:
                    self._ticket_strategy = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                pass

    def _save_ticket_strategy(self) -> None:
        import json
        self._ticket_strategy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ticket_strategy_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._ticket_strategy.items()}, f, indent=2)

    # ── Legacy compatibility shims ────────────────────────────────────────────
    # Keep old method names alive so existing code doesn't need updating
    # before the migration is complete.

    def getPositions(self):           return self.get_positions()
    def getAccountInfo(self):         return self.get_account_info()
    def get_tick_data(self, sym):     t = self.get_tick(sym); return t.__dict__ if t else None
    def get_total_floating_profit(self): return sum(p.profit for p in self.get_positions())

    def close_all_positions(self):
        for sym in {p.symbol for p in self.get_positions()}:
            self.close_position(sym)

    def close_profitable_positions(self, symbol=None):
        positions = self.get_positions()
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        results = []
        for pos in positions:
            if pos.profit > 0:
                ok = self.close_position(pos.symbol)
                results.append(f"{'✅' if ok else '❌'} {pos.symbol}")
        return "\n".join(results) if results else "No profitable positions."

    def _strategy_for(self, ticket: int) -> str:
        return self.strategy_for_ticket(ticket)

    def _log_trade_history(self, **kwargs):
        self.log_trade_history(**kwargs)

    def _mark_cooldown_public(self, symbol: str):
        self._mark_cooldown(symbol)


# ── Utility ────────────────────────────────────────────────────────────────────

def _infer_category(name: str) -> str:
    n = name.upper()
    if any(n.startswith(p) for p in ["XAU", "XAG", "XPT", "XPD"]):
        return "metals"
    if any(t in n for t in ["BTC","ETH","LTC","XBT","DOGE","ADA","SOL","XRP"]):
        return "crypto"
    if any(t in n for t in ["US30","US500","NAS","GER","UK100","JPN","AUS200","SPX","NDX","DAX","FTSE","CAC"]):
        return "indices"
    if any(t in n for t in ["OIL","NGAS","BRENT","CORN","WHEAT","COFFEE","COCOA","SUGAR"]):
        return "commodities"
    return "forex"