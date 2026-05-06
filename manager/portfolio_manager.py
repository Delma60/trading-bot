import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import pandas_ta as ta
import json
from datetime import datetime
from pathlib import Path
from .risk_manager import RiskManager
from typing import Dict

class PortfolioManager:
    """The ML-Driven Offense Engine: Learns which strategies work in which environments."""
    
    MODEL_PATH = Path("data/portfolio_meta_model.keras")
    TRAINING_DATA_PATH = Path("data/trade_history.json")
    CONFIG_PATH = Path("data/portfolio_config.json")
    
    def __init__(self, broker, strategy_manager, risk_manager: RiskManager):
        self.broker = broker
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        
        # Load the fallback config
        self.config = self._load_json(self.CONFIG_PATH, fallback={})
        self.asset_classes = self.config.get("asset_classes", {"Forex": ["EURUSD"]})
        self.strategy_mapping = self.config.get("strategy_mapping", {})
        
        self.master_watchlist = []
        for symbols in self.asset_classes.values():
            self.master_watchlist.extend(symbols)
            
        self.available_strategies = self.strategy_manager.engines.keys()
        
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
            print("[Portfolio AI]: I have enough data. Initiating first training sequence...")
            self.retrain_model()
            return True
            
        print(f"[Portfolio AI]: Only {len(history)}/50 trades logged. Using config defaults until I have enough data to learn.")
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
            print(f"[Portfolio Manager]: Failed to save config: {e}")

    def add_symbol(self, symbol: str, asset_class: str = None) -> bool:
        symbol = symbol.upper()
        if not asset_class:
            asset_class = self._infer_asset_class(symbol)

        if asset_class not in self.asset_classes:
            self.asset_classes[asset_class] = []

        if symbol in self.asset_classes[asset_class]:
            return False

        self.asset_classes[asset_class].append(symbol)
        self.master_watchlist.append(symbol)
        self.config["asset_classes"] = self.asset_classes
        self._save_config()
        return True

    def _infer_asset_class(self, symbol: str) -> str:
        symbol = symbol.upper()
        if any(symbol.startswith(prefix) for prefix in ["XAU", "XAG", "XPT", "XPD"]):
            return "Metals"
        if any(token in symbol for token in ["BTC", "ETH", "LTC", "XBT", "USDT", "DOGE"]):
            return "Crypto"
        if any(token in symbol for token in ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]):
            return "Forex"
        return "Forex"

    def _get_current_market_state(self, symbol: str) -> np.ndarray:
        """Gathers and normalizes market regime features."""
        try:
            df = self.broker.get_historical_rates(symbol, timeframe="H1", count=50)
            
            if df is None or len(df) < 50:
                return np.zeros((1, 4))
            
            # Calculate technical indicators
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_result is not None:
                df['ATR'] = atr_result
            
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_result is not None and 'ADX_14' in adx_result.columns:
                df['ADX'] = adx_result['ADX_14']
            else:
                df['ADX'] = 50.0  # Default middle value
            
            rsi_result = ta.rsi(df['close'], length=14)
            if rsi_result is not None:
                df['RSI'] = rsi_result
            
            latest = df.iloc[-1]
            
            # Extract values with defaults if missing
            atr_val = latest.get('ATR', latest['close'] * 0.01)
            adx_val = latest.get('ADX', 50.0)
            rsi_val = latest.get('RSI', 50.0)
            
            norm_atr = (atr_val / latest['close']) * 100 if latest['close'] > 0 else 0 
            norm_adx = adx_val / 100.0
            norm_rsi = rsi_val / 100.0
            norm_hour = datetime.now().hour / 24.0
            
            state = np.array([norm_atr, norm_adx, norm_rsi, norm_hour])
            state = np.nan_to_num(state)
            
            return np.expand_dims(state, axis=0)
        
        except Exception as e:
            print(f"[Portfolio Manager]: Error getting market state for {symbol}: {e}")
            return np.zeros((1, 4))

    def _assign_strategy_fallback(self, symbol: str) -> str:
        """Uses the config file if the AI isn't ready yet."""
        asset_class = next((k for k, v in self.asset_classes.items() if symbol in v), "Unknown")
        overrides = self.strategy_mapping.get("symbol_overrides", {})
        if symbol in overrides: return overrides[symbol]
        class_defaults = self.strategy_mapping.get("asset_class_defaults", {})
        if asset_class in class_defaults: return class_defaults[asset_class]
        return self.strategy_mapping.get("default", "Mean_Reversion")

    def _assign_strategy(self, symbol: str, current_state: np.ndarray) -> str:
        """Decides which strategy to use (AI if ready, Config if not)."""
        if not self.is_ai_ready:
            return self._assign_strategy_fallback(symbol)
            
        predictions = self.model.predict(current_state, verbose=0)[0]
        best_strategy_index = np.argmax(predictions)
        return self.available_strategies[best_strategy_index]

    def evaluate_portfolio_opportunities(self, risk_pct: float, stop_loss: float, max_daily_loss: float) -> list:
        allowed, reason = self.risk_manager.is_trading_allowed(max_daily_loss)
        if not allowed:
            return [f"🛑 Portfolio Halt: {reason}"]

        results = []
        for symbol in self.master_watchlist:
            # Get the state to feed the AI (and to log it if we take a trade!)
            current_state = self._get_current_market_state(symbol)
            strategy_name = self._assign_strategy(symbol, current_state)
            print(f"[Portfolio Manager]: {symbol} assigned to strategy '{strategy_name}' by AI.")
            signal = self.strategy_manager.check_signals(symbol, strategy=strategy_name)
            
            if signal and signal.get('action') != 'WAIT':
                trade_plan = self.risk_manager.calculate_safe_trade(
                    symbol=symbol, base_risk_pct=risk_pct, stop_loss_pips=stop_loss, max_daily_loss=max_daily_loss
                )
                
                if trade_plan["approved"]:
                    lots = trade_plan["lots"]
                    sl_pips = trade_plan["stop_loss_pips"]
                    
                    exec_result = self.broker.execute_trade(
                        symbol=symbol,
                        action=signal['action'],
                        lots=lots,
                        stop_loss_pips=sl_pips,
                        take_profit_pips=sl_pips * 2.0  # Example: Setting a 1:2 Risk/Reward ratio
                    )
                    
                    if exec_result["success"]:
                        ticket = exec_result["ticket"]
                        fill_price = exec_result["price"]
                        results.append(f"🟢 EXECUTED -> {symbol}: {signal['action']} | Size: {lots} | Price: {fill_price} | Ticket: #{ticket}")
                        
                        # (Optional) If you are using the AI logging, store the state here
                        self._temporary_trade_states = getattr(self, '_temporary_trade_states', {})
                        self._temporary_trade_states[symbol] = {
                            "state": current_state.tolist(), # Assuming you saved current_state earlier
                            "strategy": strategy_name
                        }
                    else:
                        reason = exec_result["reason"]
                        results.append(f"⚠️ EXECUTION FAILED -> {symbol}: {signal['action']} | Reason: {reason}")
                else:
                    results.append(f"❌ {symbol} [{strategy_name}]: Rejected. {trade_plan['reason']}")
        return results if results else ["⚠️ No high-probability entries found."]

    def log_trade_for_learning(self, symbol: str, profit: float):
        """Call this when a trade CLOSES to log the result."""
        trade_data = getattr(self, '_temporary_trade_states', {}).get(symbol)
        if not trade_data:
            return # We don't have the starting state for this trade
            
        if profit > 0:
            strategy_idx = self.available_strategies.index(trade_data["strategy"])
            label = [1 if i == strategy_idx else 0 for i in range(len(self.available_strategies))]
            
            entry = {"state": trade_data["state"], "label": label}
            
            history = self._load_json(self.TRAINING_DATA_PATH, fallback=[])
            history.append(entry)
            
            with open(self.TRAINING_DATA_PATH, "w") as f:
                json.dump(history, f)
            print(f"[Portfolio AI]: Logged winning trade for {symbol} to memory.")
            
            # Check if we should wake up the AI
            if not self.is_ai_ready and len(history) >= 50:
                self.is_ai_ready = True
                self.retrain_model()
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
        
        print(f"[Portfolio AI]: Retraining on {len(X)} historical trades...")
        self.model.fit(X, y, epochs=50, batch_size=8, verbose=1)
        self.model.save(str(self.MODEL_PATH))
        print("[Portfolio AI]: Retraining complete.")
        