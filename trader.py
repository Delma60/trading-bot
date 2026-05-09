import MetaTrader5 as mt5
import psutil
import os
import pandas as pd
import csv
import threading
from datetime import datetime, timedelta
from pathlib import Path

class Trader:
    
    def __init__(self, login="", password="", server="MetaQuotes-Demo", notify_callback=print, magic=1000):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.notify = notify_callback
        self.magic = magic
        # Centralized Execution Lock (prevents race conditions across all threads)
        self._execution_lock = threading.Lock()
        self._pending_orders = set()  # Tracks in-flight orders to avoid duplicates
        self._cooldown: dict[str, datetime] = {}
        self._cooldown_lock = threading.Lock()
        self._cooldown_seconds = 5  # default cooldown duration in seconds
        self._ticket_strategy: dict[int, str] = {}  # maps ticket → strategy name
        # Legacy lock for compatibility
        self._order_lock = self._execution_lock

    def set_cooldown(self, seconds: int):
        """Adjust cooldown duration in seconds."""
        self._cooldown_seconds = max(0, seconds)

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        """Return whether a symbol is still cooling down and remaining seconds."""
        with self._cooldown_lock:
            expiry = self._cooldown.get(symbol)
            if expiry is None:
                return False, 0.0
            remaining = (expiry - datetime.now()).total_seconds()
            if remaining <= 0:
                del self._cooldown[symbol]
                return False, 0.0
            return True, remaining

    def _mark_cooldown(self, symbol: str):
        """Mark a symbol as cooling down after a successful close."""
        if self._cooldown_seconds <= 0:
            return
        with self._cooldown_lock:
            self._cooldown[symbol] = datetime.now() + timedelta(seconds=self._cooldown_seconds)
        
    def _strategy_for(self, ticket: int) -> str:
        """Return the strategy that opened this ticket, then forget it."""
        return self._ticket_strategy.pop(ticket, "Unknown")

    def is_mt5_running(self):
        """Check if MetaTrader 5 terminal is running"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'terminal64.exe' in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
        
    def connect(self, login, password, server="MetaQuotes-Demo"):
        self.login = login
        self.password = password
        self.server = server
        
        if not self.is_mt5_running():
            self.notify("❌ MT5 terminal is not running! Start C:/Program Files/MetaTrader 5/terminal64.exe")
            return False
        
        # Try explicit path first, then automatic fallback silently
        mt5_path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        init_success = mt5.initialize(path=mt5_path, timeout=60000) if os.path.exists(mt5_path) else mt5.initialize(timeout=60000)

        if not init_success:
            self.notify(f"❌ MT5 Init failed. Code: {mt5.last_error()}")
            return False
            
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            self.notify(f"❌ Login failed. Code: {mt5.last_error()}")
            return False
        
        self.connected = True
        self.notify(f"✅ MT5 Connected: {self.login} on {self.server}")
        return True
    
    def ensure_connected(self) -> bool:
        """Safety Net: Validates connection before ANY broker action. Auto-reconnects on failure."""
        if not self.connected:
            self.notify("⚠️ Not currently connected to MT5. Connection required.")
            return False
        
        # Check if MT5 terminal is still responsive
        try:
            if mt5.terminal_info() is None:
                self.notify("⚠️ MT5 connection lost — attempting reconnect...")
                if self.login and self.password and self.server:
                    success = self.connect(self.login, self.password, self.server)
                    if success:
                        self.notify("✅ MT5 reconnected successfully.")
                    return success
                return False
        except Exception as e:
            self.notify(f"⚠️ MT5 connection check failed: {e}. Attempting reconnect...")
            if self.login and self.password and self.server:
                try:
                    success = self.connect(self.login, self.password, self.server)
                    if success:
                        self.notify("✅ MT5 reconnected successfully.")
                    return success
                except:
                    return False
            return False
        
        return True
    
    def _log_trade_history(self, action: str, symbol: str, lots: float, price: float, ticket: int, comment: str, strategy: str = "Unknown", profit: float = None):
        """Logs executed trades to a CSV file for future analytics/ML training."""
        file_path = Path("data/trade_history.csv")
        file_exists = file_path.exists()
        
        # Ensure the data directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write headers if the file is new
            if not file_exists:
                writer.writerow(["Timestamp", "Ticket", "Action", "Symbol", "Volume", "Execution_Price", "Comment", "Strategy", "Profit"])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticket,
                action,
                symbol,
                lots,
                price,
                comment,
                strategy,
                profit if profit is not None else ""
            ])
    
    def getSymbols(self):
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        symbols = mt5.symbols_get()
        return symbols

    def search_symbols(
        self,
        query: str = None,
        category: str = None,
        max_results: int = 50,
    ) -> list[dict]:
        """
        Search and filter all symbols available on the connected broker.

        Parameters
        ----------
        query    : Optional string matched against symbol name and description.
                   e.g. "USD", "GOLD", "BTC"
        category : Optional category filter — 'forex', 'metals', 'crypto',
                   'indices', 'commodities'
        max_results : Hard cap on returned results.

        Returns
        -------
        List of dicts: {name, description, category, spread_pips}
        spread_pips is None when the symbol is not currently quoted (market closed).
        """
        if not self.connected:
            return []

        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return []

        # ── Category inference rules ──────────────────────────────────────────
        def _infer_category(name: str) -> str:
            n = name.upper()
            if any(n.startswith(p) for p in ["XAU", "XAG", "XPT", "XPD"]):
                return "metals"
            if any(t in n for t in ["BTC", "ETH", "LTC", "XBT", "DOGE", "ADA", "SOL", "XRP"]):
                return "crypto"
            if any(t in n for t in ["US30", "US500", "NAS", "GER", "UK100",
                                      "JPN", "AUS200", "SPX", "NDX", "DAX",
                                      "FTSE", "CAC", "NIKKEI"]):
                return "indices"
            if any(t in n for t in ["OIL", "NGAS", "BRENT", "CORN",
                                      "WHEAT", "COFFEE", "COCOA", "SUGAR"]):
                return "commodities"
            return "forex"

        q   = query.upper().strip()   if query    else None
        cat = category.lower().strip() if category else None

        results = []

        for sym in all_symbols:
            name = sym.name
            desc = getattr(sym, "description", "") or ""
            sym_cat = _infer_category(name)

            # Category filter
            if cat and sym_cat != cat:
                continue

            # Keyword filter — match name OR description
            if q and q not in name.upper() and q not in desc.upper():
                continue

            # Live spread (fast: MT5 is local, <1ms per call)
            spread_pips = None
            try:
                tick = mt5.symbol_info_tick(name)
                info = mt5.symbol_info(name)
                if tick and info and info.point > 0:
                    spread_pts = (tick.ask - tick.bid) / info.point
                    # 5-digit forex: 10 points per pip; everything else: 1:1
                    pip_mult = 10.0 if info.digits in (5, 3) else 1.0
                    spread_pips = round(spread_pts / pip_mult, 1)
            except Exception:
                pass

            results.append({
                "name":        name,
                "description": desc,
                "category":    sym_cat,
                "spread_pips": spread_pips,
            })

            if len(results) >= max_results:
                break

        # Sort: exact name prefix matches first, then alphabetical within category
        if q:
            results.sort(key=lambda r: (0 if r["name"].startswith(q) else 1, r["name"]))
        else:
            results.sort(key=lambda r: (r["category"], r["name"]))

        return results

    def ohclv_data(self, symbol: str, timeframe: str="H1", num_bars: int=1000) -> pd.DataFrame:
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 0, num_bars)
        # FIX: Check if rates is valid before processing
        if rates is None or len(rates) == 0:
            return None
        # Set date as index
        df = pd.DataFrame(rates)
        if 'time' in df.columns:
            df = df.set_index(pd.to_datetime(df['time'], unit='s'))
            df.drop(columns=['time'], inplace=True)

        # MT5 returns volume as tick_volume or real_volume depending on broker settings.
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'real_volume' in df.columns:
                df['volume'] = df['real_volume']

        return df
        
    def get_symbol_info(self, symbol: str):
            if not self.connected:
                self.notify("Not connected to MetaTrader 5")
                return None
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                return symbol_info._asdict()
            else:
                # self.notify(f"Failed to get symbol info, error code = {mt5.last_error()}")
                return None
    def get_tick_data(self, symbol: str, num_ticks: int = 10):
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            return tick._asdict()
        else:
            # self.notify(f"Failed to get tick data, error code = {mt5.last_error()}")
            return None
    def getPositions(self):
        """Get all open positions"""
        if not self.connected:
            return None
        positions = mt5.positions_get()
        return positions
    
    def getAccountInfo(self):
        """Get account information"""
        if not self.connected:
            return None
        account = mt5.account_info()
        return account
    
    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: float = None) -> bool:
        """
        Modifies an open position's Stop Loss and/or Take Profit in MT5.
        CRITICAL: Re-verifies ticket exists inside lock to prevent "Invalid Ticket" errors
        when ProfitGuard and TrailingStopManager race on same position.
        """
        # Safety check: ensure connection before ANY broker action
        if not self.ensure_connected():
            return False

        with self._execution_lock:
            # CRITICAL: Re-verify position exists INSIDE lock
            # This prevents: ProfitGuard and TrailingStopManager both try to modify ticket simultaneously
            # → Thread A grabs lock, closes trade, Thread B re-verifies position exists INSIDE lock, sees it's gone, returns False silently
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                # Position already closed by another thread - return silently (no error)
                return False

            pos = positions[0]
            tp_value = float(new_tp) if new_tp is not None else float(pos.tp or 0.0)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": tp_value,
                "magic": self.magic,
            }

            result = mt5.order_send(request)

        if result and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
            return True
        else:
            error = getattr(result, 'comment', 'Unknown MT5 Error') if result else "Unknown MT5 Error"
            self.notify(f"[Broker] Failed to modify ticket #{ticket}: {error}")
            return False
    
    def _get_pip_multiplier(self, symbol: str) -> float:
        """
        Returns the pip-to-point multiplier for a given symbol.
        Standard Forex (5-digit): 1 pip = 10 points (multiplier = 10)
        Metals, Indices, Crypto: 1 pip = 1 point (multiplier = 1)
        """
        symbol = symbol.upper()
        if any(symbol.startswith(prefix) for prefix in ["XAU", "XAG", "XPT", "XPD"]):
            return 1.0  # Metals: 1 pip = 1 point
        if any(token in symbol for token in ["BTC", "ETH", "LTC", "XBT", "USDT", "DOGE"]):
            return 1.0  # Crypto: 1 pip = 1 point
        if "30" in symbol or "500" in symbol:  # Indices like US30, UK100
            return 1.0  # Indices: 1 pip = 1 point
        return 10.0  # Default Forex: 1 pip = 10 points
        
    def execute_trade(self, symbol: str, action: str, lots: float, stop_loss_pips: float = 0.0, take_profit_pips: float = 0.0, strategy: str = "Unknown") -> dict:
        """
        Builds and sends a live order to MetaTrader 5. All orders serialized via _execution_lock.
        """
        # Safety check: ensure connection before ANY broker action
        if not self.ensure_connected():
            return {"success": False, "reason": "Lost connection to MT5. Reconnection failed."}

        # Cooldown prevents immediate re-entry after any close
        in_cd, remaining = self.is_in_cooldown(symbol)
        if in_cd:
            return {
                "success": False,
                "reason": f"{symbol} in cooldown — {remaining:.1f}s remaining."
            }

        order_key = f"{symbol}_{action.upper()}"

        # 1. Prevent duplicate submissions for the same symbol/action
        with self._execution_lock:
            if order_key in self._pending_orders:
                return {"success": False, "reason": f"Order for {symbol} already in flight."}
            self._pending_orders.add(order_key)

        try:
            # 2. Verify the symbol is available and visible
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "reason": f"{symbol} not found on broker."}
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"success": False, "reason": f"Could not select {symbol} in Market Watch."}

            # 2. Define the Order Type and get the correct execution price
            order_type = mt5.ORDER_TYPE_BUY if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            
            if not tick:
                return {"success": False, "reason": f"Could not fetch current price for {symbol}."}
                
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            point = symbol_info.point
            pip_multiplier = self._get_pip_multiplier(symbol)

            # 3. Calculate Stop Loss and Take Profit prices securely against Spread
            sl_price = 0.0
            tp_price = 0.0

            digits = symbol_info.digits
            min_stop_level = symbol_info.trade_stops_level or 0
            spread_points = int(round((tick.ask - tick.bid) / point))

            sl_points = int(stop_loss_pips * pip_multiplier) if stop_loss_pips > 0 else 0
            tp_points = int(take_profit_pips * pip_multiplier) if take_profit_pips > 0 else 0

            safe_distance = spread_points + min_stop_level
            if sl_points > 0:
                sl_points = max(sl_points, safe_distance)
            if tp_points > 0:
                tp_points = max(tp_points, safe_distance)

            if order_type == mt5.ORDER_TYPE_BUY:
                if sl_points > 0:
                    sl_price = tick.ask - (sl_points * point)
                if tp_points > 0:
                    tp_price = tick.ask + (tp_points * point)
            elif order_type == mt5.ORDER_TYPE_SELL:
                if sl_points > 0:
                    sl_price = tick.bid + (sl_points * point)
                if tp_points > 0:
                    tp_price = tick.bid - (tp_points * point)

            # Round to the broker's supported digit precision
            if sl_price:
                sl_price = round(sl_price, digits)
            if tp_price:
                tp_price = round(tp_price, digits)

            # Sanity check: ensure stop levels are on the correct side of the price
            if sl_price and ((order_type == mt5.ORDER_TYPE_BUY and sl_price >= price) or (order_type == mt5.ORDER_TYPE_SELL and sl_price <= price)):
                return {"success": False, "reason": "Invalid Stop Loss level after normalization."}
            if tp_price and ((order_type == mt5.ORDER_TYPE_BUY and tp_price <= price) or (order_type == mt5.ORDER_TYPE_SELL and tp_price >= price)):
                return {"success": False, "reason": "Invalid Take Profit level after normalization."}

            filling_mode = symbol_info.filling_mode
            
            # Check supported modes using bitwise AND operator (1 = FOK, 2 = IOC)
            if filling_mode & 1:
                # FOK (Fill Or Kill) - Complete volume must be filled, or the order is canceled.
                type_filling = mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:
                # IOC (Immediate Or Cancel) - Fill what you can immediately, cancel the rest.
                type_filling = mt5.ORDER_FILLING_IOC
            else:
                # RETURN - Standard market execution, allowed to be partially filled and remain on the book.
                type_filling = mt5.ORDER_FILLING_RETURN
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lots),
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,          # Max slippage allowed (in points)
                "magic": self.magic,      # Unique Bot ID
                "comment": "AI Bot Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }

            # 5. Send the order - SERIALIZED via execution lock to prevent race conditions
            with self._execution_lock:
                # Re-check connection inside lock (in case connection dropped during wait)
                if not self.connected or mt5.terminal_info() is None:
                    return {"success": False, "reason": "Connection lost while waiting for execution lock."}
                
                result = mt5.order_send(request)

            # 6. Check if it succeeded and handle specific error codes
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self._ticket_strategy[result.order] = strategy   # register which strategy opened this ticket
                self._log_trade_history(
                    action=action.upper(),
                    symbol=symbol,
                    lots=lots,
                    price=result.price,
                    ticket=result.order,
                    comment=request.get("comment", ""),
                    strategy=strategy
                )
                return {"success": True, "ticket": result.order, "price": result.price}
            
            # Handle specific error conditions
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                self.notify(f"⚠️ REQUOTE on {symbol}: Ask={tick.ask}, Bid={tick.bid}. Price moved too fast. Retry.")
                return {"success": False, "reason": f"Requote at {tick.ask}/{tick.bid}. Price moved during submission."}
            
            elif result.retcode == mt5.TRADE_RETCODE_PRICE_CHANGED:
                self.notify(f"⚠️ PRICE_CHANGED on {symbol}: Slippage detected. Requested {price}, market moved.")
                return {"success": False, "reason": f"Price changed before execution. Slippage occurred."}
            
            else:
                self.notify(f"❌ Order rejected. MT5 Code: {result.retcode}, Comment: {result.comment}")
                return {"success": False, "reason": f"Order rejected. MT5 Code: {result.retcode}, {result.comment}"}
        finally:
            with self._execution_lock:
                self._pending_orders.discard(order_key)

    
    def get_historical_rates(self, symbol: str, timeframe: str = "H1", count: int = 50):
        """Fetches historical OHLCV data for a symbol."""
        if not self.connected:
            return None
            
        # Map your string timeframe to MT5 timeframes
        tf_mapping = {
             "M1": mt5.TIMEFRAME_M1,   "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,   "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,   "W1": mt5.TIMEFRAME_W1,
    "MN": mt5.TIMEFRAME_MN1,
        }
        mt5_tf = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
        if mt5_tf is None:
            self.notify(f"[Broker] Unknown timeframe '{timeframe}', defaulting to H1")
            mt5_tf = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None:
            self.notify(f"[Broker] Failed to fetch rates for {symbol}")
            return None
            
        df = pd.DataFrame(rates)
        return df
    
    def getBalance(self):
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        account_info = mt5.account_info()
        if account_info is not None:
            return account_info.balance
    
    def get_daily_realized_profit(self):
        """Get total realized profit/loss for today."""
        if not self.connected:
            return 0.0
            
        from datetime import datetime
        
        # Calculate timestamps for the start of today and right now
        today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        today_end = int(datetime.now().timestamp())
        
        # Use today_end instead of the invalid mt5.TIME_CURRENT
        deals = mt5.history_deals_get(today_start, today_end)
        
        if deals is None:
            return 0.0
            
        return sum(deal.profit for deal in deals if deal.profit is not None)
    
    def get_total_floating_profit(self):
        """Get total floating profit from open positions."""
        if not self.connected:
            return 0.0
        positions = mt5.positions_get()
        if positions is None:
            return 0.0
        return sum(position.profit for position in positions if position.profit is not None)
    
    def close_all_positions(self):
        """Close all open positions."""
        if not self.connected:
            self.notify("Cannot close positions: Not connected to MT5.")
            return

        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            self.notify("No open positions to close.")
            return

        self.notify(f"Attempting to close {len(positions)} open positions...")

        with self._order_lock:
            for position in positions:
                symbol = position.symbol
                symbol_info = mt5.symbol_info(symbol)

                if symbol_info is None:
                    self.notify(f"❌ Failed to get symbol info for {symbol}")
                    continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    self.notify(f"❌ Could not get live tick price for {symbol}, skipping...")
                    continue

                # To close a BUY, we must SELL at the BID price. To close a SELL, we must BUY at the ASK price.
                if position.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask

                # Determine correct filling mode based on broker limits
                filling_mode = symbol_info.filling_mode
                if filling_mode & 1:
                    type_filling = mt5.ORDER_FILLING_FOK
                elif filling_mode & 2:
                    type_filling = mt5.ORDER_FILLING_IOC
                else:
                    type_filling = mt5.ORDER_FILLING_RETURN

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Bot manual close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": type_filling,
                }

                result = mt5.order_send(request)

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.notify(f"❌ Failed to close {symbol} (Ticket #{position.ticket}). MT5 Code: {result.retcode}, {result.comment}")
                else:
                    self._log_trade_history(
                        action="CLOSE",
                        symbol=symbol,
                        lots=position.volume,
                        price=result.price,
                        ticket=result.order,
                        comment=f"Profit: {position.profit}",
                        strategy=self._strategy_for(position.ticket),
                        profit=position.profit,
                    )
                    self._mark_cooldown(symbol)
                    self.notify(f"✅ Successfully closed {symbol} (Ticket #{position.ticket}) at {result.price}")

    def close_position(self, symbol: str):
        """Close open position(s) for a specific symbol. Serialized via _execution_lock."""
        # Safety check: ensure connection before ANY broker action
        if not self.ensure_connected():
            self.notify("Cannot close position: Lost connection to MT5.")
            return False

        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            self.notify(f"❌ No open positions found for {symbol}.")
            return False

        self.notify(f"Attempting to close {len(positions)} position(s) for {symbol}...")

        success = True
        with self._execution_lock:
            # Re-verify connection inside lock
            if not self.connected or mt5.terminal_info() is None:
                self.notify("❌ Connection lost while waiting for execution lock.")
                return False
            
            # Re-fetch positions inside lock (in case they changed during wait)
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                self.notify(f"❌ No open positions found for {symbol} (may have been closed by another thread).")
                return False
            
            for position in positions:
                symbol_info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)

                if tick is None or symbol_info is None:
                    self.notify(f"❌ Could not close {symbol} due to missing symbol data.")
                    success = False
                    continue

                if position.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask

                filling_mode = symbol_info.filling_mode
                if filling_mode & 1:
                    type_filling = mt5.ORDER_FILLING_FOK
                elif filling_mode & 2:
                    type_filling = mt5.ORDER_FILLING_IOC
                else:
                    type_filling = mt5.ORDER_FILLING_RETURN

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": self.magic,
                    "comment": "Bot manual close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": type_filling,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.notify(f"❌ Failed to close {symbol} (Ticket #{position.ticket}). MT5 Code: {result.retcode}")
                    success = False
                else:
                    self._log_trade_history(
                        action="CLOSE",
                        symbol=symbol,
                        lots=position.volume,
                        price=result.price,
                        ticket=result.order,
                        comment=f"Profit: {position.profit}",
                        strategy=self._strategy_for(position.ticket),
                        profit=position.profit,
                    )
                    self._mark_cooldown(symbol)
                    self.notify(f"✅ Successfully closed {symbol} (Ticket #{position.ticket}) at {result.price}")

        return success
    def partial_close_position(self, ticket: int, symbol: str,
        close_ratio: float = 0.5) -> dict:
        if not 0.1 <= close_ratio <= 0.9:
            return {"success": False, "reason": "close_ratio must be 0.1–0.9"}
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "reason": f"Ticket #{ticket} not found"}
        pos = positions[0]
        volume_to_close = round(pos.volume * close_ratio, 2)
        sym_info = mt5.symbol_info(symbol)
        if sym_info and volume_to_close < sym_info.volume_min:
            return {"success": False, "reason": "Volume below broker minimum"}
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if pos.type == 0 else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
            "volume": volume_to_close, "type": order_type,
            "position": ticket, "price": price, "deviation": 20,
            "magic": self.magic, "comment": "Partial close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self._log_trade_history("PARTIAL_CLOSE", symbol, volume_to_close,
                                    result.price, result.order, "", profit=None)
            return {"success": True, "ticket": result.order,
                    "volume_closed": volume_to_close}
        return {"success": False, "reason": result.comment}
    def close_profitable_positions(self, symbol: str = None):
        """Close only currently profitable open positions."""
        if not self.connected:
            return "Cannot close trades: Not connected to MT5."

        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return "No open positions found."

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
            if not positions:
                return f"No open positions found for {symbol}."

        profitable_positions = [p for p in positions if p.profit is not None and p.profit > 0]
        if not profitable_positions:
            return "No profitable positions found to close."

        responses = []
        with self._execution_lock:
            for position in profitable_positions:
                pos_symbol = position.symbol
                symbol_info = mt5.symbol_info(pos_symbol)
                if symbol_info is None:
                    responses.append(f"Failed to close {pos_symbol}: could not fetch symbol info.")
                    continue

                tick = mt5.symbol_info_tick(pos_symbol)
                if tick is None:
                    responses.append(f"Failed to close {pos_symbol}: could not fetch current tick price.")
                    continue

                if position.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask

                filling_mode = symbol_info.filling_mode
                if filling_mode & 1:
                    type_filling = mt5.ORDER_FILLING_FOK
                elif filling_mode & 2:
                    type_filling = mt5.ORDER_FILLING_IOC
                else:
                    type_filling = mt5.ORDER_FILLING_RETURN

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos_symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Close profitable trade",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": type_filling,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    responses.append(f"❌ Failed to close {pos_symbol} (Ticket #{position.ticket}): {result.comment}")
                else:
                    self._log_trade_history(
                        action="CLOSE",
                        symbol=pos_symbol,
                        lots=position.volume,
                        price=result.price,
                        ticket=result.order,
                        comment=f"Profit: {position.profit}",
                        strategy=self._strategy_for(position.ticket),
                        profit=position.profit,
                    )
                    self._mark_cooldown(pos_symbol)
                    responses.append(f"✅ Closed {pos_symbol} (Ticket #{position.ticket}) at {result.price}")

        return "\n".join(responses)