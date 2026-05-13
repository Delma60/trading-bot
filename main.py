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
from manager.profile_manager import profile as _profile
from manager.position_monitor import PositionMonitor
# --- Optimizer imports ---
from manager.self_optimizer import SelfOptimizer
from manager.auto_optimizer import AutoOptimizer

# Global shutdown flag for graceful termination
shutdown_event = threading.Event()
current_agent_listener = None
_scan_lock = threading.Lock()  # Prevent overlapping scan cycles

def _default_agent_notify(msg: str, priority: str = "normal"):
    timestamp = datetime.now().strftime("%H:%M:%S")
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


def handle_external_close(ticket: int, symbol: str, profit: float, close_price: float, direction: str, lots: float, open_price: float):
    """
    Fired by PositionMonitor when MT5 closes a trade via SL/TP or manual app intervention.
    """
    strategy_name = broker._strategy_for(ticket)
    
    broker._log_trade_history(
        action="CLOSE_SL_TP",
        symbol=symbol,
        lots=lots,
        price=close_price,
        ticket=ticket,
        comment=f"External close (SL/TP) | Profit: {profit}",
        strategy=strategy_name,
        profit=profit
    )
    
    if profit < 0:
        risk_manager.record_loss(symbol)
    else:
        risk_manager.record_win(symbol)
        
    portfolio_manager.log_trade_for_learning(ticket=ticket, profit=profit)
    
    status_icon = "🟢" if profit > 0 else "🔴"
    agent_notify(
        f"{status_icon} External Close Detected -> {symbol} (Ticket #{ticket}) closed at {close_price} | PnL: ${profit:.2f}",
        priority="trade_executed"
    )
    
def autonomous_scanner(portfolio_manager: PortfolioManager, scan_interval_seconds: int = 3, notify=agent_notify):
    """
    Background scanner. Re-reads profile each cycle so runtime changes
    (e.g. 'change risk to 0.5%') take effect immediately.
    """
    notify(f"🟢 Real-time Market Watch started. Scanning every {scan_interval_seconds} seconds.")

    while not shutdown_event.is_set():
        if not _scan_lock.acquire(blocking=False):
            if shutdown_event.wait(timeout=scan_interval_seconds):
                break
            continue

        try:
            # FIX #18: Re-read profile on every cycle so chat-driven changes
            # (risk, symbols, daily goal) are picked up immediately.
            r  = _profile.risk()
            symbols = _profile.symbols()
            risk_pct = r.risk_pct
            stop_loss = r.stop_loss_pips
            max_daily_loss = r.max_daily_loss
            daily_target = r.daily_goal
            session_target = r.take_profit_pips * len(symbols)

            today_profit = portfolio_manager.broker.get_daily_realized_profit()
            floating_profit = portfolio_manager.broker.get_total_floating_profit()

            if today_profit + floating_profit >= daily_target:
                notify(f"🎉 Daily Goal of ${daily_target} reached! Closing all trades.")
                portfolio_manager.broker.close_all_positions()

                for symbol in symbols:
                    portfolio_manager.strategy_manager.continuous_learning_routine(symbol)

                notify("💤 Neural Net optimized. Sleeping until tomorrow.")
                # FIX #24: Sleep until midnight rather than a fixed 86400 s offset
                # so we resume at the next trading day open, not 24h from goal hit.
                from datetime import date, timedelta
                import time as _time
                midnight = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
                secs_to_midnight = (midnight - datetime.now()).total_seconds()
                if shutdown_event.wait(timeout=max(secs_to_midnight, 1)):
                    break
                continue

            if floating_profit >= session_target:
                notify(f"✅ Session Goal of ${session_target} reached! Closing basket and starting a new cycle.")
                portfolio_manager.broker.close_all_positions()

            results = portfolio_manager.evaluate_portfolio_opportunities(
                risk_pct=risk_pct,
                stop_loss=stop_loss,
                max_daily_loss=max_daily_loss
            )

            for result in results:
                priority = "trade_executed" if "EXECUTED" in result else "critical" if "🚑" in result or "FATAL" in result else "normal"
                notify(result, priority=priority)

            if shutdown_event.wait(timeout=scan_interval_seconds):
                notify("🚑 Shutdown signal received. Scanner stopping gracefully...")
                break

        except Exception as e:
            notify(f"⚠️ Error during autonomous scan: {e}")
            if shutdown_event.wait(60):
                break
        finally:
            _scan_lock.release()

    notify("✅ Background Scanner stopped.")

def signal_handler(signum, frame):
    agent_notify("🛑 Shutdown signal received (Ctrl+C or termination).", priority="critical")
    shutdown_event.set()


def _resolve_symbols(broker, symbols: list[str]) -> list[str]:
    from manager.symbol_registry import SymbolRegistry

    if symbols:
        return symbols

    if broker.connected:
        try:
            registry = SymbolRegistry(broker)
            result = []
            result.extend(registry.get_universe("forex")[:5])
            result.extend(registry.get_universe("metals")[:2])
            result.extend(registry.get_universe("indices_us")[:2])
            result.extend(registry.get_universe("crypto")[:2])
            if result:
                agent_notify(
                    f"profile.json has no symbols — defaulting to "
                    f"{len(result)} broker-sourced symbols across asset classes."
                )
                return result
        except Exception as exc:
            agent_notify(f"⚠️ Could not query broker for default symbols: {exc}")

    agent_notify(
        "⚠️ Broker unavailable and _profile.json has no trading_symbols. "
        "Defaulting to EURUSD."
    )
    return ["EURUSD"]


if __name__ == "__main__":
    agent_notify("Initializing Quantitative Trading System...")

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

    symbols = _resolve_symbols(broker, _profile.symbols())
    cache = LocalCache(broker, symbols, notify_callback=agent_notify)
    cache.warm_up()
    cache.start()

    strategy_manager = StrategyManager(broker, cache=cache, notify_callback=agent_notify)

    # FIX #2: max_open_trades must NOT equal len(symbols) — that bypasses all
    # position limits and scales exposure linearly with watchlist size.
    # Read from profile (default 2) and cap at a safe ceiling.
    _broker_cfg = _profile.broker()
    _safe_max_trades = min(_broker_cfg.max_open_trades, 5)  # hard ceiling of 5
    risk_manager = RiskManager(
        broker,
        cache=cache,
        max_open_trades=_safe_max_trades,
        min_margin_level=150.0,
        notify_callback=agent_notify,
    )

    portfolio_manager = PortfolioManager(broker, strategy_manager, risk_manager, cache=cache, notify_callback=agent_notify)
    
    scanner_thread = threading.Thread(
        target=autonomous_scanner,
        args=(portfolio_manager, _profile.scanner().interval_seconds, agent_notify)
    )
    scanner_thread.daemon = False
    scanner_thread.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bot = ARIA(
        intents_filepath="intents.json", 
        broker=broker, 
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager
    )
    current_agent_listener = bot.receive_system_alert

    # FIX #5/#9: Optimizers started here (on main thread, before chat loop)
    # rather than as dead class-level code that never ran.
    self_optimizer = SelfOptimizer(strategy_manager, broker, notify_callback=agent_notify)
    auto_optimizer = AutoOptimizer(strategy_manager, notify_callback=agent_notify)
    self_optimizer.start()
    auto_optimizer.start()

    try:
        bot.start_chat()
    except KeyboardInterrupt:
        agent_notify("Keyboard interrupt detected.")
        shutdown_event.set()
    except Exception as e:
        import traceback
        agent_notify(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        shutdown_event.set()
    finally:
        agent_notify("Initiating graceful shutdown sequence...")
        shutdown_event.set()
        
        agent_notify("Waiting for background scanner to stop...")
        scanner_thread.join(timeout=5)
        if scanner_thread.is_alive():
            agent_notify("⚠️ Scanner thread did not stop within timeout.")
        
        if 'cache' in locals() and cache is not None:
            agent_notify("Saving cache to disk and stopping background refresh...")
            cache.stop()

        if broker.connected:
            agent_notify("Disconnecting from MetaTrader 5...")
            try:
                broker.disconnect()
                agent_notify("✅ Broker disconnected successfully.")
            except Exception as e:
                agent_notify(f"⚠️ Error disconnecting broker: {e}")
        
        agent_notify("👋 Trading bot shutdown complete. Goodbye!\n")
        sys.exit(0)