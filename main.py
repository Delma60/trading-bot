import json 
import time
import threading
import signal
import sys
from trader import Trader
from strategies.strategy_manager import StrategyManager
from manager.risk_manager import RiskManager
from manager.portfolio_manager import PortfolioManager
from chat import Chatbot
from pathlib import Path

# Global shutdown flag for graceful termination
shutdown_event = threading.Event()
current_agent_listener = None

def _default_agent_notify(msg: str, priority: str = "normal"):
    prefix = ""
    if priority == "critical":
        prefix = "⚠️ "
    elif priority == "trade_executed":
        prefix = "🟢 "
    elif priority == "normal":
        prefix = ""
    print(f"\n[Agent]{prefix}: {msg}")

def agent_notify(msg: str, priority: str = "normal"):
    """
    Centralized communication channel for the agent.
    If a Chatbot listener is registered, route notifications through the bot's inbox.
    """
    if current_agent_listener is not None:
        current_agent_listener(msg, priority)
    else:
        _default_agent_notify(msg, priority)

def autonomous_scanner(portfolio_manager: PortfolioManager, scan_interval_minutes: int = 15, notify=agent_notify):
    """
    This runs in the background forever. It wakes up, scans the market, 
    executes trades if it finds any, and goes back to sleep.
    """
    notify(f"🟢 Background Scanner started. Waking up every {scan_interval_minutes} minutes.")
    
    # Define your global risk rules here (or load them from a config file)
    profile_path = Path("data/profile.json")

    while not shutdown_event.is_set():
        try:
            # 1. Wait for the interval (convert minutes to seconds) but check shutdown flag
            if shutdown_event.wait(scan_interval_minutes * 60):
                notify("🛑 Shutdown signal received. Scanner stopping gracefully...")
                break
            
            config = {}
            if profile_path.exists():
                with open(profile_path, "r") as f:
                    config = json.load(f)
                    
            risk_pct = config.get("risk_percentage", 1.0)
            stop_loss = config.get("stop_loss", 20.0)
            max_daily_loss = config.get("max_daily_loss", 50.0)
            
            # 2. Wake up and scan!
            notify("Waking up to scan markets... 🔎")
            results = portfolio_manager.evaluate_portfolio_opportunities(
                risk_pct=risk_pct,
                stop_loss=stop_loss,
                max_daily_loss=max_daily_loss
            )
            
            # 3. Send the results through the agent
            for result in results:
                priority = "trade_executed" if "EXECUTED" in result else "critical" if "🛑" in result or "FATAL" in result else "normal"
                notify(result, priority=priority)
                
        except Exception as e:
            notify(f"⚠️ Error during autonomous scan: {e}")
            # Sleep for a minute before retrying to prevent error spam
            if shutdown_event.wait(60):
                break
    
    notify("✅ Background Scanner stopped.")

def signal_handler(signum, frame):
    """
    Handle Ctrl+C (SIGINT) and termination signals gracefully.
    """
    agent_notify("🛑 Shutdown signal received (Ctrl+C or termination).", priority="critical")
    shutdown_event.set()

if __name__ == "__main__":
    agent_notify("Initializing Quantitative Trading System...")

    # 1. Initialize the core API connection
    broker = Trader(notify_callback=agent_notify)

    # 2. Initialize the Engines
    strategy_manager = StrategyManager(broker, notify_callback=agent_notify)
    risk_manager = RiskManager(broker, max_open_trades=3, min_margin_level=150.0, notify_callback=agent_notify)

    # 3. Initialize the Portfolio Manager
    portfolio_manager = PortfolioManager(broker, strategy_manager, risk_manager, notify_callback=agent_notify)

    # 4. Start the Background Scanner Thread
    # Set daemon=False so we can gracefully join it on exit
    scanner_thread = threading.Thread(
        target=autonomous_scanner,
        args=(portfolio_manager, 10, agent_notify)
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
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager
    )
    current_agent_listener = bot.receive_system_alert

    # 7. Run chatbot with proper exception handling and cleanup
    try:
        bot.start_chat()
    except KeyboardInterrupt:
        agent_notify("Keyboard interrupt detected.")
        shutdown_event.set()
    except Exception as e:
        agent_notify(f"❌ Unexpected error: {e}")
        shutdown_event.set()
    finally:
        # === GRACEFUL SHUTDOWN SEQUENCE ===
        agent_notify("Initiating graceful shutdown sequence...")
        
        # 1. Signal scanner thread to stop
        shutdown_event.set()
        
        # 2. Wait for scanner thread to finish (with timeout)
        agent_notify("Waiting for background scanner to stop...")
        scanner_thread.join(timeout=5)
        if scanner_thread.is_alive():
            agent_notify("⚠️ Scanner thread did not stop within timeout.")
        
        # 3. Disconnect from broker
        if broker.connected:
            agent_notify("Disconnecting from MetaTrader 5...")
            try:
                broker.disconnect()
                agent_notify("✅ Broker disconnected successfully.")
            except Exception as e:
                agent_notify(f"⚠️ Error disconnecting broker: {e}")
        
        # 4. Final goodbye message
        agent_notify("👋 Trading bot shutdown complete. Goodbye!\n")
        sys.exit(0)