import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import pandas_ta as ta
import json
from datetime import datetime
from pathlib import Path
import threading
from .risk_manager import RiskManager
from typing import Dict
from manager.market_sessions import MarketSessionManager
from manager.risk_manager import TradeGatekeeper, CorrelationGuard
from strategies.strategy_manager import OHLCVCache
from manager.market_sessions import MarketSessionManager
from manager.profile_manager import profile as _profile
from manager.correlation_matrix import PortfolioHeatCheck

class PortfolioManager:
    """The ML-Driven Offense Engine: Learns which strategies work in which environments."""
    
    MODEL_PATH = Path("data/portfolio_meta_model.keras")
    TRAINING_DATA_PATH = Path("data/trade_history.json")
    PROFILE_PATH = Path("data/profile.json")  # Single source of truth
    
    def __init__(self, broker, strategy_manager, risk_manager: RiskManager, cache=None, notify_callback=print):
        self.broker = broker
        b = _profile.broker()
        s = _profile.sessions()
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.notify = notify_callback
        self.gate = TradeGatekeeper(
            max_spread_pips       = b.spread_tolerance_pips,
            avoid_asian_session   = s.avoid_asian_session,
            avoid_friday_close    = s.avoid_friday_close,
        )
        
        self.allocation_limits = {
            "Forex": 0.5,   # Max 50% of portfolio in Forex
            "Crypto": 0.3,  # Max 30% of portfolio in Crypto
            "Metals": 0.2,  # Max 20% of portfolio in Metals
        }
        self.corr_guard = CorrelationGuard(max_shared_legs=2)
        
        p = _profile.portfolio()
        self.asset_classes    = p.asset_classes
        self.strategy_mapping = p.strategy_mapping
        # FIX: Use a set to prevent duplicate symbols in the master watchlist
        initial_symbols = _profile.symbols()
        asset_symbols = [s for sublist in self.asset_classes.values() for s in sublist]
        self.master_watchlist = list(set(initial_symbols + asset_symbols))

        self.available_strategies = list(self.strategy_manager.engines.keys())

        # Use provided cache or get from strategy manager
        self._ohlcv_cache = cache or getattr(self.strategy_manager, '_ohlcv_cache', OHLCVCache(ttl_seconds=60))

        # Thread-safe lock for model predictions (prevents race conditions between main and background threads)
        # FIX: Added a dedicated lock for trade state management
        self._state_lock = threading.Lock()
        self._temporary_trade_states: dict[int, dict] = {}
        self._model_lock = threading.Lock()
        self._temporary_trade_states: dict[int, dict] = {}
        
        # AI Initialization
        self.model = self._load_or_build_model()
        self.is_ai_ready = self._check_ai_readiness()

    def _load_json(self, filepath: Path, fallback: dict) -> dict:
        if filepath.exists():
            try:
                with filepath.open("r") as f:
                    return json.load(f)
            except Exception: pass
        return fallback

    def _check_ai_readiness(self) -> bool:
        """Checks if we have enough historical data for the AI to make decisions."""
        if self.MODEL_PATH.exists():
            return True # Model has already been trained and saved
            
        history = self._load_json(self.TRAINING_DATA_PATH, fallback=[])
        if len(history) >= 50:
            # [Silent] Model training initiated from historical data
            self.retrain_model()
            return True
            
        # [Silent] Waiting for sufficient historical data before training
        return False

    def _load_or_build_model(self):
        if self.MODEL_PATH.exists():
            return load_model(str(self.MODEL_PATH))
            
        model = Sequential([
            Dense(16, input_shape=(4,), activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(self.available_strategies), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _save_config(self):
        try:
            with self.CONFIG_PATH.open("w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.notify(f"[Portfolio Manager]: Failed to save config: {e}")

    # def add_symbol(self, symbol: str, asset_class: str = None) -> bool:
    #     symbol = symbol.upper()
    #     if not asset_class:
    #         asset_class = self._infer_asset_class(symbol)

    #     if asset_class not in self.asset_classes:
    #         self.asset_classes[asset_class] = []

    #     if symbol in self.asset_classes[asset_class]:
    #         return False

    #     self.asset_classes[asset_class].append(symbol)
    #     self.master_watchlist.append(symbol)
    #     self.config["asset_classes"] = self.asset_classes
    #     self._save_config()
    #     return True
    
    def add_symbol(self, symbol: str, asset_class: str = None) -> bool:
        symbol = symbol.upper()
        # FIX: Delegate persistence directly to the single source of truth
        if not asset_class:
            asset_class = self._infer_asset_class(symbol)

        if asset_class not in self.asset_classes:
            self.asset_classes[asset_class] = []

        if symbol in self.asset_classes[asset_class]:
            return False

        self.asset_classes[asset_class].append(symbol)
        self.master_watchlist.append(symbol)
        # If you have a config dict, persist here if needed
        # self.config["asset_classes"] = self.asset_classes
        # self._save_config()
        return True

    def _assign_strategy(self, symbol: str, current_state: np.ndarray) -> str:
        """Decides which strategy to use (AI if ready, Config if not)."""
        if not self.is_ai_ready:
            return self._assign_strategy_fallback(symbol)
            
        predictions = self._predict_strategy_safe(current_state)[0]
        best_strategy_index = np.argmax(predictions)
        return self.available_strategies[best_strategy_index]

    def evaluate_portfolio_opportunities(self, risk_pct: float, stop_loss: float, max_daily_loss: float, dry_run: bool = False) -> list:
        results = []
        session_manager = MarketSessionManager()
        tradeable_symbols, closed_symbols = session_manager.filter_tradeable_symbols(self.master_watchlist)

        if not tradeable_symbols:
            return [
                "⚠️ No tradeable symbols in your portfolio right now. "
                "Crypto is open 24/7 if you want active exposure."
            ]

        if closed_symbols:
            results.append(
                f"⏳ Skipping {len(closed_symbols)} closed symbols. "
                f"{len(tradeable_symbols)} symbols remain tradeable."
            )

        # FIX: Risk Scaling Logic
        # Use total watchlist size for risk distribution, not just currently open markets.
        total_portfolio_capacity = len(self.master_watchlist)

        account_info = self.broker.getAccountInfo()
        if account_info is None:
            return results

        positions = self.broker.getPositions() or []
        symbol_counts = {symbol: 0 for symbol in tradeable_symbols}
        for p in positions:
            if p.symbol in symbol_counts:
                symbol_counts[p.symbol] += 1

        prioritized_symbols = sorted(tradeable_symbols, key=lambda s: symbol_counts.get(s, 0))

        for symbol in prioritized_symbols:
            allowed, reason = self.risk_manager.is_trading_allowed(
                symbol=symbol, 
                max_daily_loss=max_daily_loss,
                portfolio_size=total_portfolio_capacity  # Updated to use total capacity
            )

            if not allowed:
                results.append(f"⚠️ {symbol}: Skipped. {reason}")
                continue

            ok, corr_reason = self.corr_guard.check(symbol, self.broker)

            if not ok:
                results.append(f"⚠️ {symbol}: {corr_reason}")
                continue
            if self.risk_manager.is_loss_paused(symbol):
                results.append(f"⏸ {symbol}: loss-streak pause active, skipping.")
                continue

            current_state = self._get_current_market_state(symbol)
            strategy_name = self._assign_strategy(symbol, current_state)

            # Reduced threshold from 0.65 to 0.50 to allow valid multi-timeframe signals
            MIN_SIGNAL_CONFIDENCE = _profile.scanner().min_signal_confidence   
            signal = self.strategy_manager.check_signals(symbol, use_ensemble=True)
            if (signal
                    and signal.get('action') != 'WAIT'
                    and signal.get('confidence', 0.0) >= MIN_SIGNAL_CONFIDENCE):
                gate_ok, gate_reason = self.gate.gate(symbol, self.broker)
                if not gate_ok:
                    results.append(f"⚠️ {symbol}: Trade gate blocked entry. {gate_reason}")
                    continue
                
                action = signal['action']

                reentry_sys = getattr(self.risk_manager, 'reentry_system', None)
                if reentry_sys and symbol in reentry_sys.stopped_out_trades:
                    record = reentry_sys.stopped_out_trades[symbol]
                    time_elapsed = (datetime.now() - record["time"]).total_seconds() / 3600
                        
                    self.heat_check = PortfolioHeatCheck(
                        cache            = self._ohlcv_cache,
                        lookback         = 100,
                        max_correlation  = 0.80,
                        ttl_seconds      = 60,
                    )
                    
                    if not record["re_entered"] and time_elapsed <= 4.0:
                        open_syms = [p.symbol for p in self.broker.getPositions() or []]
                        heat_ok, heat_reason, _ = self.heat_check.check(symbol, open_syms)
                        if not heat_ok:
                            results.append(f"⚠️ {symbol}: {heat_reason}")
                            continue
                        tick = self.broker.get_tick_data(symbol)
                        if tick:
                            current_price = tick.get("ask") if action == "BUY" else tick.get("bid")
                            if current_price:
                                if not reentry_sys.check_reentry_validity(symbol, current_price, action):
                                    results.append(f"⚠️ {symbol}: Blocked by Smart Re-Entry rules (price hasn't swept SL/recovered or wrong direction).")
                                    continue
                                else:
                                    self.notify(f"🔄 Smart Re-Entry approved for {symbol}! Price swept liquidity and recovered.")

                trade_plan = self.risk_manager.calculate_safe_trade(
                    symbol=symbol, base_risk_pct=risk_pct, stop_loss_pips=stop_loss, max_daily_loss=max_daily_loss, portfolio_size=total_portfolio_capacity
                )
                
                if trade_plan["approved"]:
                    if dry_run:
                        results.append(
                            f"📈 {symbol}: {signal['action']} | confidence {signal.get('confidence', 0.0):.0%} | strategy {signal.get('strategy_used', strategy_name)}"
                        )
                        continue

                    lots = trade_plan["lots"]
                    sl_pips = trade_plan["stop_loss_pips"]
                    
                    exec_result = self.broker.execute_trade(
                        symbol=symbol,
                        action=signal['action'],
                        lots=lots,
                        stop_loss_pips=sl_pips,
                        take_profit_pips=sl_pips * 2.0,
                        strategy=signal.get('strategy_used', strategy_name)
                    )
                    
                    if exec_result.get("success"):
                        ticket = exec_result.get("ticket")
                        fill_price = exec_result.get("price")
                        results.append(f"🟢 EXECUTED -> {symbol}: {signal['action']} | Size: {lots} | Price: {fill_price} | Ticket: #{ticket}")
                        if reentry_sys:
                            reentry_sys.mark_reentered(symbol)
                        self._temporary_trade_states[ticket] = {
                            "state": current_state.flatten().tolist(),
                            "strategy": strategy_name,
                            "action": signal["action"],
                            "symbol": symbol,
                            "article_ids": [signal.get("article_id")] if signal.get("article_id") else signal.get("article_ids", []),
                            "entry_price": fill_price,
                        }
                    else:
                        error_reason = exec_result.get("reason", "unknown error")

                        if "margin" in error_reason.lower():
                            new_lots = self.risk_manager.calculate_micro_lot()
                            self.notify(f"⚠️ [Scanner]: Margin rejection for {symbol}. Attempting recovery execution at {new_lots} micro-lots.")
                            
                            retry_exec = self.broker.execute_trade(
                                symbol=symbol,
                                action=signal['action'],
                                lots=new_lots,
                                stop_loss_pips=sl_pips,
                                take_profit_pips=sl_pips * 2.0,
                                strategy=signal.get('strategy_used', strategy_name)
                            )
                            
                            if retry_exec.get("success"):
                                ticket = retry_exec.get("ticket")
                                results.append(f"🟢 RECOVERY EXECUTED -> {symbol}: {signal['action']} | Size: {new_lots} | Ticket: #{ticket}")
                                if reentry_sys:
                                    reentry_sys.mark_reentered(symbol)
                                self._temporary_trade_states[ticket] = {
                                    "state": current_state.flatten().tolist(),
                                    "strategy": strategy_name,
                                    "action": signal["action"],
                                    "symbol": symbol,
                                    "article_ids": [signal.get("article_id")] if signal.get("article_id") else signal.get("article_ids", []),
                                    "entry_price": fill_price,
                                }
                            else:
                                results.append(f"❌ RECOVERY FAILED -> {symbol}: {retry_exec.get('reason', 'unknown error')}")
                        elif "market closed" in error_reason.lower():
                            results.append(f"⏳ MARKET CLOSED -> {symbol}")
                        else:
                            results.append(f"⚠️ EXECUTION FAILED -> {symbol}: {signal['action']} | Reason: {error_reason}")
                else:
                    results.append(f"❌ {symbol} [{strategy_name}]: Rejected. {trade_plan['reason']}")
        return results if results else ["⚠️ No high-probability entries found."]
    
    def log_trade_for_learning(self, ticket: int = None, symbol: str = None, profit: float = None):
        """Call this when a trade CLOSES to log the result."""
        if ticket is not None:
            trade_data = self._temporary_trade_states.pop(ticket, None)
        elif symbol is not None:
            matching = [t for t, data in self._temporary_trade_states.items() if data.get("symbol") == symbol]
            if len(matching) == 1:
                trade_data = self._temporary_trade_states.pop(matching[0])
            else:
                trade_data = None
        else:
            trade_data = None

        if not trade_data or profit is None:
            return

        fv = np.array(trade_data["state"])
        action = trade_data.get("action", "BUY")
        self.strategy_manager.record_trade_outcome(fv, action, profit)

        # Feedback loop for news classifier if trade was news-based
        article_ids = trade_data.get("article_ids", [])
        entry_price = trade_data.get("entry_price")
        symbol = trade_data.get("symbol")
        
        if article_ids and entry_price is not None and symbol:
            try:
                tick_data = self.broker.get_tick_data(symbol)
                if tick_data:
                    current_price = tick_data.get("bid") if action == "SELL" else tick_data.get("ask")
                    if current_price:
                        for article_id in article_ids:
                            self.evaluate_news_impact(article_id, entry_price, current_price, action)
            except:
                pass  # Silently fail if price fetch doesn't work
        
    def evaluate_news_impact(self, article_id: str, entry_price: float, current_price: float, action: str = "BUY"):
        """
        Evaluates the impact of news on price movement and feeds back to the news classifier.
        Reconstructs directional movement based on action and realized PnL.
        """
        price_delta_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Only label if the market actually moved significantly (≥0.05%)
        if abs(price_delta_pct) >= 0.05:
            # Reconstruct directional movement based on action
            if action == "BUY":
                direction = "UP" if price_delta_pct > 0 else "DOWN"
            else:  # SELL
                direction = "DOWN" if price_delta_pct > 0 else "UP"
            
            # Use the realized price delta as the label signal
            delta_proxy = abs(price_delta_pct)
            
            # Inject the label back into the LSTM training pool
            try:
                news_strategy = self.strategy_manager.engines.get("News_Trading") or self.strategy_manager.strategies.get("news_trading")
                if news_strategy and hasattr(news_strategy, "classifier"):
                    news_strategy.classifier.inject_price_label(
                        article_id=article_id, 
                        label=direction, 
                        delta=delta_proxy
                    )
            except:
                pass  # Silently fail if classifier not available

    def get_portfolio_health(self) -> str:
        """Generates a quick health report for the Chatbot."""
        account = self.broker.getAccountInfo()
        positions = self.broker.getPositions()
        open_count = len(positions) if positions else 0
        
        if not account:
            return "Unable to assess portfolio health."
            
        daily_pnl = account.equity - account.balance
        health = "🟢 Healthy" if daily_pnl >= 0 else "🟡 Drawdown"
        if account.margin_level and account.margin_level < 200:
            health = "🔴 High Risk (Low Margin)"

        return f"{health} | PnL Today: ${daily_pnl:,.2f} | Exposure: {open_count}/{self.risk_manager.max_open_trades} Trades"
    def retrain_model(self):
        history = self._load_json(self.TRAINING_DATA_PATH, fallback=[])
        if len(history) < 50: return
            
        X = np.array([item["state"][0] for item in history])
        y = np.array([item["label"] for item in history])
        
        # [Silent] Model retraining on historical data
        # Use lock to prevent race conditions during training
        with self._model_lock:
            self.model.fit(X, y, epochs=50, batch_size=8, verbose=0)  # Changed verbose=1 to verbose=0
            self.model.save(str(self.MODEL_PATH))
        