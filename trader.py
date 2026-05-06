import MetaTrader5 as mt5
import psutil
import os

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
        

    def getPositions(self):
        if not self.connected:
            print("Not connected to MetaTrader 5")
            return None
        positions = mt5.positions_get()
        return positions
    
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