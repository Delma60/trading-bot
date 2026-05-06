import time
import threading
import signal
import sys
from trader import Trader
from strategies.strategy_manager import StrategyManager
from manager.risk_manager import RiskManager
from manager.portfolio_manager import PortfolioManager
from chat import Chatbot

# Global shutdown flag for graceful termination
shutdown_event = threading.Event()

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

    while not shutdown_event.is_set():
        try:
            # 1. Wait for the interval (convert minutes to seconds) but check shutdown flag
            if shutdown_event.wait(scan_interval_minutes * 60):
                print("\n[Scanner]: 🛑 Shutdown signal received. Scanner stopping gracefully...")
                break
            
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
            if shutdown_event.wait(60):
                break
    
    print("[System]: ✅ Background Scanner stopped.")

def signal_handler(signum, frame):
    """
    Handle Ctrl+C (SIGINT) and termination signals gracefully.
    """
    print("\n[System]: 🛑 Shutdown signal received (Ctrl+C or termination).")
    shutdown_event.set()

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
    # Set daemon=False so we can gracefully join it on exit
    scanner_thread = threading.Thread(
        target=autonomous_scanner,
        args=(portfolio_manager, 10)
    )
    scanner_thread.daemon = False
    scanner_thread.start()

    # 5. Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # 6. Boot up the Chatbot on the Main Thread
    bot = Chatbot(
        intents_filepath="intents.json", 
        broker=broker, 
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager
    )

    # 7. Run chatbot with proper exception handling and cleanup
    try:
        bot.start_chat()
    except KeyboardInterrupt:
        print("\n[System]: Keyboard interrupt detected.")
        shutdown_event.set()
    except Exception as e:
        print(f"[System]: ❌ Unexpected error: {e}")
        shutdown_event.set()
    finally:
        # === GRACEFUL SHUTDOWN SEQUENCE ===
        print("\n[System]: Initiating graceful shutdown sequence...")
        
        # 1. Signal scanner thread to stop
        shutdown_event.set()
        
        # 2. Wait for scanner thread to finish (with timeout)
        print("[System]: Waiting for background scanner to stop...")
        scanner_thread.join(timeout=5)
        if scanner_thread.is_alive():
            print("[System]: ⚠️ Scanner thread did not stop within timeout.")
        
        # 3. Disconnect from broker
        if broker.connected:
            print("[System]: Disconnecting from MetaTrader 5...")
            try:
                broker.disconnect()
                print("[System]: ✅ Broker disconnected successfully.")
            except Exception as e:
                print(f"[System]: ⚠️ Error disconnecting broker: {e}")
        
        # 4. Final goodbye message
        print("[System]: 👋 Trading bot shutdown complete. Goodbye!\n")
        sys.exit(0)