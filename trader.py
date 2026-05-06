import MetaTrader5 as mt5
import psutil
import os
import pandas as pd

class Trader:
    
    def __init__(self, login="", password="", server="MetaQuotes-Demo"):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
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
        
        # Check if MT5 is running
        if not self.is_mt5_running():
            print("MetaTrader 5 terminal is not running!")
            print("Please start MT5 terminal and try again.")
            print("You can find it in: C:/Program Files/MetaTrader 5/terminal64.exe")
            return False
        
        # Try to initialize MT5 with different approaches
        print("Attempting to connect to MetaTrader 5...")
        
        # First try: with explicit path and longer timeout
        try:
            mt5_path = "C:/Program Files/MetaTrader 5/terminal64.exe"
            if os.path.exists(mt5_path):
                if not mt5.initialize(path=mt5_path, timeout=120000):
                    print("Failed with explicit path, trying without path...")
                    # Second try: let MT5 find the terminal automatically
                    if not mt5.initialize(timeout=120000):
                        print("initialize() failed, error code =", mt5.last_error())
                        print("Please ensure:")
                        print("1. MetaTrader 5 terminal is running")
                        print("2. MT5 is installed in the default location")
                        print("3. No firewall/antivirus is blocking the connection")
                        return False
            else:
                print(f"MT5 not found at {mt5_path}, trying automatic detection...")
                if not mt5.initialize(timeout=120000):
                    print("initialize() failed, error code =", mt5.last_error())
                    return False
                    
        except Exception as e:
            print(f"Exception during initialization: {e}")
            return False
            
        print("MT5 initialized successfully, attempting login...")
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            print("Login failed, error code =", mt5.last_error())
            return False
        
        self.connected = True
        print("Successfully connected to MetaTrader 5")
        return True
    
    def getSymbols(self):
        if not self.connected:
            print("Not connected to MetaTrader 5")
            return None
        symbols = mt5.symbols_get()
        return symbols
    
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

        # 3. Calculate Stop Loss and Take Profit prices
        # Note: We multiply pips by 10 to convert to points for standard forex pairs.
        sl_price = 0.0
        tp_price = 0.0
        
        if order_type == mt5.ORDER_TYPE_BUY:
            if stop_loss_pips > 0: sl_price = price - (stop_loss_pips * 10 * point)
            if take_profit_pips > 0: tp_price = price + (take_profit_pips * 10 * point)
        elif order_type == mt5.ORDER_TYPE_SELL:
            if stop_loss_pips > 0: sl_price = price + (stop_loss_pips * 10 * point)
            if take_profit_pips > 0: tp_price = price - (take_profit_pips * 10 * point)

        # 4. Build the MT5 Request Dictionary
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
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        # 5. Send the order!
        result = mt5.order_send(request)
        
        # 6. Check if it succeeded
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": f"Order rejected. MT5 Code: {result.retcode}, {result.comment}"}

        return {"success": True, "ticket": result.order, "price": result.price}

    def getPositions(self):
        if not self.connected:
            print("Not connected to MetaTrader 5")
            return None
        positions = mt5.positions_get()
        return positions
    
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
            print(f"[Broker] Failed to fetch rates for {symbol}")
            return None
            
        return pd.DataFrame(rates)
    
    def getBalance(self):
        if not self.connected:
            print("Not connected to MetaTrader 5")
            return None
        account_info = mt5.account_info()
        if account_info is not None:
            return account_info.balance
        else:
            print("Failed to get account info, error code =", mt5.last_error())
            return None