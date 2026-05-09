import os
# Must be set BEFORE any other imports to silence TensorFlow C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json 
import time
import threading
import signal
import sys
from datetime import datetime
from trader import Trader
from strategies.strategy_manager import StrategyManager
from manager.local_cache import LocalCache
from manager.risk_manager import RiskManager
from manager.portfolio_manager import PortfolioManager
from chat import ARIA
from pathlib import Path

# Global shutdown flag for graceful termination
shutdown_event = threading.Event()
current_agent_listener = None
_scan_lock = threading.Lock()  # Prevent overlapping scan cycles

def _default_agent_notify(msg: str, priority: str = "normal"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Format directly to your required standard
    print(f"[{timestamp}] [Bot]: {msg}")

def agent_notify(msg: str, priority: str = "normal"):
    """
    Centralized communication channel for the agent.
    If a Chatbot listener is registered, route notifications through the bot's inbox.
    """
    if current_agent_listener is not None:
        current_agent_listener(msg, priority)
    else:
        _default_agent_notify(msg, priority)

def autonomous_scanner(portfolio_manager: PortfolioManager, scan_interval_seconds: int = 3, notify=agent_notify):
    """
    This runs in the background forever. It wakes up, scans the market, 
    executes trades if it finds any, and goes back to sleep.
    """
    notify(f"🟢 Real-time Market Watch started. Scanning every {scan_interval_seconds} seconds.")
    
    # Define your global risk rules here (or load them from a config file)
    profile_path = Path("data/profile.json")

    while not shutdown_event.is_set():
        # Prevent overlapping scan cycles — skip if previous scan still running
        if not _scan_lock.acquire(blocking=False):
            if shutdown_event.wait(timeout=scan_interval_seconds):
                break
            continue
        
        try:
            config = {}
            if profile_path.exists():
                with open(profile_path, "r") as f:
                    config = json.load(f)
                    
            risk_pct = config.get("risk_percentage", 1.0)
            stop_loss = config.get("stop_loss", 20.0)
            max_daily_loss = config.get("max_daily_loss", 50.0)
            daily_target = config.get("daily_goal", 10.0)
            session_target = config.get("target_profit", 1.0) * len(config.get("trading_symbols", []))
            
            # 1. Check total realized profit for today
            today_profit = portfolio_manager.broker.get_daily_realized_profit()
            
            # 2. Check floating profit of open trades (the current session)
            floating_profit = portfolio_manager.broker.get_total_floating_profit()
            
            # 3. Have we hit the daily goal?
            if today_profit + floating_profit >= daily_target:
                notify(f"🎉 Daily Goal of ${daily_target} reached! Closing all trades.")
                portfolio_manager.broker.close_all_positions()
                
                # +++ TRIGGER CONTINUOUS LEARNING HERE +++
                for symbol in config.get("trading_symbols", []):
                    portfolio_manager.strategy_manager.continuous_learning_routine(symbol)
                
                notify("💤 Neural Net optimized. Sleeping until tomorrow.")
                time.sleep(86400)  # 24 hours
                continue
            
            # 4. Have we hit the session goal?
            if floating_profit >= session_target:
                notify(f"✅ Session Goal of ${session_target} reached! Closing basket and starting a new cycle.")
                portfolio_manager.broker.close_all_positions()
                # Continue to next scan
            
            # 5. Scan portfolio for opportunities (silent operation)
            results = portfolio_manager.evaluate_portfolio_opportunities(
                risk_pct=risk_pct,
                stop_loss=stop_loss,
                max_daily_loss=max_daily_loss
            )
            
            # 6. Send the results through the agent
            for result in results:
                priority = "trade_executed" if "EXECUTED" in result else "critical" if "🛑" in result or "FATAL" in result else "normal"
                notify(result, priority=priority)

            # 7. Wait for the next scan or a shutdown signal.
            if shutdown_event.wait(timeout=scan_interval_seconds):
                notify("🛑 Shutdown signal received. Scanner stopping gracefully...")
                break
                
        except Exception as e:
            notify(f"⚠️ Error during autonomous scan: {e}")
            # Sleep for a minute before retrying to prevent error spam
            if shutdown_event.wait(60):
                break
        finally:
            _scan_lock.release()
    
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
    credentials_path = Path("data/credentials.json")
    credentials = {}
    if credentials_path.exists():
        try:
            credentials = json.loads(credentials_path.read_text())
        except Exception as exc:
            agent_notify(f"⚠️ Failed to read credentials: {exc}")

    connected = False
    if credentials:
        connected = broker.connect(
            credentials.get("login", ""),
            credentials.get("password", ""),
            credentials.get("server", "MetaQuotes-Demo"),
        )

    if not connected:
        agent_notify("⚠️ Broker not connected. LocalCache will still warm up from disk if available.")

    # 2. Load profile symbols and initialize shared cache
    profile_path = Path("data/profile.json")
    cfg = {}
    if profile_path.exists():
        try:
            cfg = json.loads(profile_path.read_text())
        except Exception as exc:
            agent_notify(f"⚠️ Failed to read profile config: {exc}")

    symbols = cfg.get("trading_symbols", ["EURUSD"])
    cache = LocalCache(broker, symbols, notify_callback=agent_notify)
    cache.warm_up()
    cache.start()

    # 3. Initialize the Engines
    strategy_manager = StrategyManager(broker, cache=cache, notify_callback=agent_notify)
    risk_manager = RiskManager(broker, cache=cache, max_open_trades=len(symbols), min_margin_level=150.0, notify_callback=agent_notify)

    # 4. Initialize the Portfolio Manager
    portfolio_manager = PortfolioManager(broker, strategy_manager, risk_manager, cache=cache, notify_callback=agent_notify)

    # 5. Start the Background Scanner Thread
    # Set daemon=False so we can gracefully join it on exit
    scanner_thread = threading.Thread(
        target=autonomous_scanner,
        args=(portfolio_manager, 3, agent_notify)
    )
    scanner_thread.daemon = False
    scanner_thread.start()

    # 5. Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # 6. Boot up ARIA on the Main Thread
    bot = ARIA(
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
        import traceback
        agent_notify(f"❌ Unexpected error: {e}")
        traceback.print_exc()  # Print full stack trace for debugging
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
        
        # 3. Stop cache refresh loop
        if 'cache' in locals() and cache is not None:
            agent_notify("Saving cache to disk and stopping background refresh...")
            cache.stop()

        # 4. Disconnect from broker
        if broker.connected:
            agent_notify("Disconnecting from MetaTrader 5...")
            try:
                broker.disconnect()
                agent_notify("✅ Broker disconnected successfully.")
            except Exception as e:
                agent_notify(f"⚠️ Error disconnecting broker: {e}")
        
        # 5. Final goodbye message
        agent_notify("👋 Trading bot shutdown complete. Goodbye!\n")
        sys.exit(0)