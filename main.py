import time
import threading
from trader import Trader
from strategies.strategy_manager import StrategyManager
from manager.risk_manager import RiskManager
from manager.portfolio_manager import PortfolioManager
from chat import Chatbot

def autonomous_scanner(portfolio_manager: PortfolioManager, scan_interval_minutes: int = 15):
    """
    This runs in the background forever. It wakes up, scans the market, 
    executes trades if it finds any, and goes back to sleep.
    """
    print(f"\n[System]: 🟢 Background Scanner started. Waking up every {scan_interval_minutes} minutes.")
    
    # Define your global risk rules here (or load them from a config file)
    BASE_RISK_PCT = 1.0
    STOP_LOSS_PIPS = 20.0
    MAX_DAILY_LOSS = 50.0

    while True:
        try:
            # 1. Wait for the interval (convert minutes to seconds)
            time.sleep(scan_interval_minutes * 60)
            
            # 2. Wake up and scan!
            print("\n[Scanner]: Waking up to scan markets... 🔎")
            results = portfolio_manager.evaluate_portfolio_opportunities(
                risk_pct=BASE_RISK_PCT,
                stop_loss=STOP_LOSS_PIPS,
                max_daily_loss=MAX_DAILY_LOSS
            )
            
            # 3. Print the results to the terminal so the user sees it
            for result in results:
                print(f"[Scanner]: {result}")
                
        except Exception as e:
            print(f"\n[Scanner]: ⚠️ Error during autonomous scan: {e}")
            # Sleep for a minute before retrying to prevent error spam
            time.sleep(60)

if __name__ == "__main__":
    print("Initializing Quantitative Trading System...")

    # 1. Initialize the core API connection
    broker = Trader()

    # 2. Initialize the Engines
    strategy_manager = StrategyManager(broker)
    risk_manager = RiskManager(broker, max_open_trades=3, min_margin_level=150.0)

    # 3. Initialize the Portfolio Manager
    portfolio_manager = PortfolioManager(broker, strategy_manager, risk_manager)

    # 4. Start the Background Scanner Thread
    # We set daemon=True so that when you type "quit" in the chatbot, this background thread dies with it.
    scanner_thread = threading.Thread(
        target=autonomous_scanner, 
        args=(portfolio_manager, 15) # Scan every 15 minutes
    )
    scanner_thread.daemon = True 
    scanner_thread.start()

    # 5. Boot up the Chatbot on the Main Thread
    bot = Chatbot(
        intents_filepath="intents.json", 
        broker=broker, 
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager
    )

    # This will block the main thread and wait for your input, 
    # but the scanner_thread is now happily running in the background!
    bot.start_chat()