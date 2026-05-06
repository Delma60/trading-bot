import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import pandas_ta as ta
import json
from datetime import datetime
from pathlib import Path
from risk_manager import RiskManager
from typing import Dict

class DeepPortfolioManager:
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
            
        self.available_strategies = ["Mean_Reversion", "Trend_Following", "Breakout_Strategy", "Momentum_Strategy"]
        
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

    def _get_current_market_state(self, symbol: str) -> np.ndarray:
        """Gathers and normalizes market regime features."""
        df = self.broker.get_historical_rates(symbol, timeframe="H1", count=50)
        
        if df is None or len(df) < 50:
            return np.zeros((1, 4))
            
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx_df['ADX_14']
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        latest = df.iloc[-1]
        
        norm_atr = (latest['ATR'] / latest['close']) * 100 if latest['close'] > 0 else 0 
        norm_adx = latest['ADX'] / 100.0
        norm_rsi = latest['RSI'] / 100.0
        norm_hour = datetime.now().hour / 24.0
        
        state = np.array([norm_atr, norm_adx, norm_rsi, norm_hour])
        state = np.nan_to_num(state)
        
        return np.expand_dims(state, axis=0)

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
            
            signal = self.strategy_manager.check_signals(symbol, strategy=strategy_name)
            
            if signal and signal.get('action') != 'WAIT':
                trade_plan = self.risk_manager.calculate_safe_trade(
                    symbol=symbol, base_risk_pct=risk_pct, stop_loss_pips=stop_loss, max_daily_loss=max_daily_loss
                )
                
                if trade_plan["approved"]:
                    lots = trade_plan["lots"]
                    
                    # Store the state temporarily so we can log it when the trade closes
                    # In a real bot, you'd attach this state array to the ticket ID!
                    self._temporary_trade_states = getattr(self, '_temporary_trade_states', {})
                    self._temporary_trade_states[symbol] = {
                        "state": current_state.tolist(),
                        "strategy": strategy_name
                    }
                    
                    brain = "🤖 AI Select" if self.is_ai_ready else "⚙️ Config Select"
                    results.append(f"{brain} -> {symbol} [{strategy_name}]: {signal['action']} | Size: {lots}")
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

    def retrain_model(self):
        history = self._load_json(self.TRAINING_DATA_PATH, fallback=[])
        if len(history) < 50: return
            
        X = np.array([item["state"][0] for item in history])
        y = np.array([item["label"] for item in history])
        
        print(f"[Portfolio AI]: Retraining on {len(X)} historical trades...")
        self.model.fit(X, y, epochs=50, batch_size=8, verbose=1)
        self.model.save(str(self.MODEL_PATH))
        print("[Portfolio AI]: Retraining complete.")