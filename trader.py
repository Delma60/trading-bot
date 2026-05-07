import MetaTrader5 as mt5
import psutil
import os
import pandas as pd

class Trader:
    
    def __init__(self, login="", password="", server="MetaQuotes-Demo", notify_callback=print):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.notify = notify_callback
        
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
    
    def getSymbols(self):
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        symbols = mt5.symbols_get()
        return symbols
    def ohclv_data(self, symbol: str, timeframe: str="H1", num_bars: int=1000) -> pd.DataFrame:
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 0, num_bars)
        # set date as index
        df = pd.DataFrame(rates)
        df = df.set_index(pd.to_datetime(df['time'], unit='s'))
        df.drop(columns=['time'], inplace=True)
        return df
        
    def get_symbol_info(self, symbol: str):
            if not self.connected:
                self.notify("Not connected to MetaTrader 5")
                return None
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                return symbol_info._asdict()
            else:
                self.notify(f"Failed to get symbol info, error code = {mt5.last_error()}")
                return None
    def get_tick_data(self, symbol: str, num_ticks: int = 10):
        if not self.connected:
            self.notify("Not connected to MetaTrader 5")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            return tick._asdict()
        else:
            self.notify(f"Failed to get tick data, error code = {mt5.last_error()}")
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
        
    def execute_trade(self, symbol: str, action: str, lots: float, stop_loss_pips: float = 0.0, take_profit_pips: float = 0.0) -> dict:
        """
        Builds and sends a live order to MetaTrader 5.
        """
        if not self.connected:
            return {"success": False, "reason": "Not connected to MT5."}

        # 1. Verify the symbol is available and visible
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
            "magic": 234000,          # Unique Bot ID (helps you identify bot trades vs manual)
            "comment": "AI Bot Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling, # Immediate or Cancel
        }

        # 5. Send the order!
        result = mt5.order_send(request)
        
        # 6. Check if it succeeded
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": f"Order rejected. MT5 Code: {result.retcode}, {result.comment}"}

        return {"success": True, "ticket": result.order, "price": result.price}
    
    def get_historical_rates(self, symbol: str, timeframe: str = "H1", count: int = 50):
        """Fetches historical OHLCV data for a symbol."""
        if not self.connected:
            return None
            
        # Map your string timeframe to MT5 timeframes
        tf_mapping = {
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        mt5_tf = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
        
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None:
            self.notify(f"[Broker] Failed to fetch rates for {symbol}")
            return None
            
        df = pd.DataFrame(rates)
        df.to_csv(f"data/symbols/{symbol}_{timeframe}_data.csv", index=False)
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
        today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        deals = mt5.history_deals_get(today_start, mt5.TIME_CURRENT)
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
            
            # Check if MT5 actually closed it
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.notify(f"❌ Failed to close {symbol} (Ticket #{position.ticket}). MT5 Code: {result.retcode}, {result.comment}")
            else:
                self.notify(f"✅ Successfully closed {symbol} (Ticket #{position.ticket}) at {result.price}")

    def close_position(self, symbol: str):
        """Close open position for a specific symbol."""
        if not self.connected:
            self.notify("Cannot close position: Not connected to MT5.")
            return False

        # Only get positions for this specific symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            self.notify(f"❌ No open positions found for {symbol}.")
            return False

        self.notify(f"Attempting to close {len(positions)} position(s) for {symbol}...")
        
        success = True
        for position in positions:
            symbol_info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            
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
                "magic": 234000,
                "comment": "Bot manual close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.notify(f"❌ Failed to close {symbol} (Ticket #{position.ticket}). MT5 Code: {result.retcode}")
                success = False
            else:
                self.notify(f"✅ Successfully closed {symbol} (Ticket #{position.ticket}) at {result.price}")
        
        return success