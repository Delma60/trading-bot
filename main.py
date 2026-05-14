"""
main.py — ARIA entry point (updated for BrokerManager abstraction layer)

Changes from previous version
------------------------------
- `Trader` replaced by `BrokerManager` everywhere
- `BrokerManager.from_credentials()` handles credential loading + auto-fallback
- `execute_trade_legacy()` shim keeps ActionExecutor dict-format checks working
- Everything else is unchanged: scanner, risk manager, portfolio manager, ARIA
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import time
import threading
import signal
import sys
from datetime import datetime
from pathlib import Path

# ── Broker abstraction (replaces `from trader import Trader`) ─────────────────
from broker import BrokerManager

from strategies.strategy_manager import StrategyManager
from manager.local_cache import LocalCache
from manager.risk_manager import RiskManager
from manager.portfolio_manager import PortfolioManager
from chat import ARIA
from manager.profile_manager import profile as _profile
from manager.position_monitor import PositionMonitor
from manager.self_optimizer import SelfOptimizer
from manager.auto_optimizer import AutoOptimizer

# ── Globals ───────────────────────────────────────────────────────────────────

shutdown_event           = threading.Event()
current_agent_listener   = None
_scan_lock               = threading.Lock()


# ── Logging ───────────────────────────────────────────────────────────────────

def _default_agent_notify(msg: str, priority: str = "normal"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [Bot]: {msg}")


def agent_notify(msg: str, priority: str = "normal"):
    if current_agent_listener is not None:
        current_agent_listener(msg, priority)
    else:
        _default_agent_notify(msg, priority)


# ── External-close handler ────────────────────────────────────────────────────

def handle_external_close(
    ticket: int, symbol: str, profit: float, close_price: float,
    direction: str, lots: float, open_price: float,
):
    strategy_name = broker._strategy_for(ticket)

    broker._log_trade_history(
        action   = "CLOSE_SL_TP",
        symbol   = symbol,
        lots     = lots,
        price    = close_price,
        ticket   = ticket,
        comment  = f"External close (SL/TP) | Profit: {profit}",
        strategy = strategy_name,
        profit   = profit,
    )

    if profit < 0:
        risk_manager.record_loss(symbol)
    else:
        risk_manager.record_win(symbol)

    portfolio_manager.log_trade_for_learning(ticket=ticket, profit=profit)
    auto_optimizer.on_trade_closed(symbol, profit)

    icon = "🟢" if profit > 0 else "🔴"
    agent_notify(
        f"{icon} External Close → {symbol} (#{ticket}) @ {close_price} | "
        f"PnL: ${profit:.2f}",
        priority="trade_executed",
    )


# ── Background scanner ────────────────────────────────────────────────────────

def autonomous_scanner(
    portfolio_manager: PortfolioManager,
    scan_interval_seconds: int = 3,
    notify=agent_notify,
):
    notify(f"🟢 Real-time Market Watch started. Scanning every {scan_interval_seconds}s.")

    while not shutdown_event.is_set():
        if not _scan_lock.acquire(blocking=False):
            if shutdown_event.wait(timeout=scan_interval_seconds):
                break
            continue

        try:
            r              = _profile.risk()
            symbols        = _profile.symbols()
            risk_pct       = r.risk_pct
            stop_loss      = r.stop_loss_pips
            max_daily_loss = r.max_daily_loss
            daily_target   = r.daily_goal
            session_target = r.take_profit_pips * len(symbols)

            today_profit   = portfolio_manager.broker.get_daily_realized_profit()
            floating_profit = portfolio_manager.broker.get_total_floating_profit()

            if today_profit + floating_profit >= daily_target:
                notify(f"🎉 Daily Goal ${daily_target} reached! Closing all.")
                portfolio_manager.broker.close_all_positions()
                for symbol in symbols:
                    portfolio_manager.strategy_manager.continuous_learning_routine(symbol)
                notify("💤 Neural Net optimized. Sleeping until tomorrow.")
                from datetime import date, timedelta
                import time as _time
                midnight = datetime.combine(
                    date.today() + timedelta(days=1), datetime.min.time()
                )
                secs = (midnight - datetime.now()).total_seconds()
                if shutdown_event.wait(timeout=max(secs, 1)):
                    break
                continue

            if floating_profit >= session_target:
                notify(f"✅ Session Goal ${session_target} reached!")
                portfolio_manager.broker.close_all_positions()

            results = portfolio_manager.evaluate_portfolio_opportunities(
                risk_pct=risk_pct,
                stop_loss=stop_loss,
                max_daily_loss=max_daily_loss,
            )
            for result in results:
                priority = (
                    "trade_executed" if "EXECUTED" in result
                    else "critical"  if ("🚑" in result or "FATAL" in result)
                    else "normal"
                )
                notify(result, priority=priority)

            if shutdown_event.wait(timeout=scan_interval_seconds):
                notify("🚑 Shutdown signal received — scanner stopping.")
                break

        except Exception as exc:
            notify(f"⚠️ Scanner error: {exc}")
            if shutdown_event.wait(60):
                break
        finally:
            _scan_lock.release()

    notify("✅ Background Scanner stopped.")


# ── Signal handlers ───────────────────────────────────────────────────────────

def signal_handler(signum, frame):
    agent_notify("🛑 Shutdown signal received.", priority="critical")
    shutdown_event.set()


# ── Symbol bootstrap ──────────────────────────────────────────────────────────

def _resolve_symbols(broker: BrokerManager, symbols: list[str]) -> list[str]:
    from manager.symbol_registry import SymbolRegistry

    if symbols:
        return symbols

    if broker.connected:
        try:
            registry = SymbolRegistry(broker)
            result = []
            result.extend(registry.get_universe("forex")[:5])
            result.extend(registry.get_universe("metals")[:2])
            result.extend(registry.get_universe("stocks")[:2])
            result.extend(registry.get_universe("crypto")[:2])
            if result:
                agent_notify(
                    f"profile.json has no symbols — defaulting to "
                    f"{len(result)} broker-sourced symbols."
                )
                return result
        except Exception as exc:
            agent_notify(f"⚠️ Could not query broker for symbols: {exc}")

    agent_notify("⚠️ No symbols found — defaulting to EURUSD.")
    return ["EURUSD"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent_notify("Initializing Quantitative Trading System…")

    # ── 1. Broker ─────────────────────────────────────────────────────────────
    broker = BrokerManager.from_credentials(notify_callback=agent_notify)

    # Attempt connection (from_credentials already stored creds internally)
    connected = broker.connect()

    if not connected:
        agent_notify(
            "⚠️ Broker not connected. "
            "LocalCache will warm up from disk if available."
        )

    # ── 2. Cache & strategy layer ─────────────────────────────────────────────
    symbols = _resolve_symbols(broker, _profile.symbols())
    cache   = LocalCache(broker, symbols, notify_callback=agent_notify)
    cache.warm_up()
    cache.start()

    strategy_manager = StrategyManager(broker, cache=cache, notify_callback=agent_notify)

    # ── 3. Risk & portfolio ───────────────────────────────────────────────────
    _broker_cfg     = _profile.broker()
    _safe_max_trades = min(_broker_cfg.max_open_trades, 5)

    risk_manager = RiskManager(
        broker,
        cache           = cache,
        max_open_trades = _safe_max_trades,
        min_margin_level = 150.0,
        notify_callback = agent_notify,
    )

    portfolio_manager = PortfolioManager(
        broker, strategy_manager, risk_manager,
        cache           = cache,
        notify_callback = agent_notify,
    )

    # ── 4. Background scanner ─────────────────────────────────────────────────
    scanner_thread = threading.Thread(
        target  = autonomous_scanner,
        args    = (portfolio_manager, _profile.scanner().interval_seconds, agent_notify),
        daemon  = False,
    )
    scanner_thread.start()

    # ── 5. Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── 6. ARIA chat ──────────────────────────────────────────────────────────
    bot = ARIA(
        intents_filepath  = "intents.json",
        broker            = broker,
        strategy_manager  = strategy_manager,
        portfolio_manager = portfolio_manager,
        risk_manager      = risk_manager,
    )
    current_agent_listener = bot.receive_system_alert

    # ── 7. Optimizers ─────────────────────────────────────────────────────────
    self_optimizer = SelfOptimizer(strategy_manager, broker, notify_callback=agent_notify)
    auto_optimizer = AutoOptimizer(strategy_manager, notify_callback=agent_notify)
    bot._auto_optimizer = auto_optimizer
    self_optimizer.start()
    auto_optimizer.start()

    # ── 8. Main loop ──────────────────────────────────────────────────────────
    try:
        bot.start_chat()
    except KeyboardInterrupt:
        agent_notify("Keyboard interrupt.")
        shutdown_event.set()
    except Exception as exc:
        import traceback
        agent_notify(f"❌ Unexpected error: {exc}")
        traceback.print_exc()
        shutdown_event.set()
    finally:
        shutdown_event.set()

        if "scanner_thread" in dir() and scanner_thread.is_alive():
            scanner_thread.join(timeout=5)

        if "cache" in dir():
            cache.stop()
        if "self_optimizer" in dir():
            self_optimizer.stop()
        if "auto_optimizer" in dir():
            auto_optimizer.stop()
        if "bot" in dir():
            bot.shutdown()
        if "broker" in dir():
            broker.disconnect()

        print("[System] Core runtime terminated.")
        agent_notify("👋 Trading bot shutdown complete.\n")
        sys.exit(0)