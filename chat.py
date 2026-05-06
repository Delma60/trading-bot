import os
import json
import pickle
import random
import re
import getpass
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional

from strategies.strategy_manager import StrategyManager
from manager.portfolio_manager import PortfolioManager

from trader import Trader


class Chatbot:
    """Handles NLP, Model Training, and Chat interface."""
    
    # Define file paths centrally
    PROFILE_FILE = Path("data/profile.json")
    HISTORY_FILE = Path("data/history.json")
    STATS_FILE = Path("data/stats.json")
    DATA_PICKLE = Path("data/data.pickle")
    MODEL_FILE = Path("data/chatbot_model.keras")

    def __init__(self, 
                intents_filepath: str, 
                broker: Trader, 
                strategy_manager: StrategyManager,
                portfolio_manager: PortfolioManager
            ):
        self.stemmer = LancasterStemmer()
        self.intents_filepath = Path(intents_filepath)
        self.broker = broker  

        self.user_anchor = None
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.last_intent = None  
        self.active_suggestion = None  

        # Trading configuration
        self.trading_symbols = []
        self.target_profit = 0.0
        self.stop_loss = 0.0
        self.risk_percentage = 0.0
        self.max_daily_loss = 0.0
        self.preferred_timeframes = []

        # Action mappings linking intents to MT5 methods
        self.action_mappings = {
            "active_positions": lambda inp: self._format_positions_data(),
            "account_summary": lambda inp: self._format_account_data(),
            "trading_symbols": lambda inp: self._format_symbols_data(),
            "trading_settings": lambda inp: self._format_settings_data(),
            "update_config": lambda inp: self.setup_trading_config(),
            "bulk_scan": lambda inp: self._execute_bulk_scan()
        }

        self.words = []
        self.labels = []
        self.training = []
        self.output = []
        self.model = None
        self.data = None

        self._load_intents()
        self._process_data()
        self._build_model()

    # --- IO Helpers ---

    def _read_json(self, filepath: Path, default: Any = None) -> Any:
        """Helper to read JSON files safely."""
        if not filepath.exists():
            return default if default is not None else {}
        try:
            with filepath.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default if default is not None else {}

    def _write_json(self, filepath: Path, data: Any):
        """Helper to write JSON files safely."""
        with filepath.open("w") as f:
            json.dump(data, f, indent=4)

    def _get_validated_input(self, prompt: str, cast_type: type, validation_func: Callable[[Any], bool], error_msg: str) -> Any:
        """Helper for robust terminal input validation."""
        while True:
            raw_input = input(prompt).strip()
            if not raw_input:
                continue
            try:
                # Handle comma-separated lists differently
                if cast_type == list:
                    val = [item.strip().upper() for item in raw_input.split(',') if item.strip()]
                else:
                    val = cast_type(raw_input)
                
                if validation_func(val):
                    return val
                print(f"[Bot]: {error_msg}")
            except ValueError:
                print(f"[Bot]: Please enter a valid {cast_type.__name__}.")

    # --- Configuration ---
    def setup_trading_config(self):
        """Conversational setup for trading configuration."""
        print("\n" + "="*50)
        print("🤖 TRADING BOT SETUP")
        print("="*50)
        print("[Bot]: Let's configure your trading preferences!")
        print("[Bot]: You can tell me your plan all at once.")
        print("       (e.g., 'I want to trade EURUSD on H1. Risk 1%, make $10 daily, stop loss $5, max daily loss $20')\n")

        user_input = input("You: ").strip()
        
        # Extract everything we can from their sentence
        parsed = self._parse_conversational_config(user_input)
        
        self.trading_symbols = parsed["symbols"]
        self.preferred_timeframes = parsed["timeframes"]
        self.risk_percentage = parsed["risk"]
        self.target_profit = parsed["target_profit"]
        self.stop_loss = parsed["stop_loss"]
        self.max_daily_loss = parsed["max_daily_loss"]

        print("\n[Bot]: Got it. Checking if I need any more details...")

        # Fill in the blanks for anything they missed
        if not self.trading_symbols:
            self.trading_symbols = self._get_validated_input(
                "[Bot]: Which currency pairs or symbols do you want to trade? (e.g. EURUSD): ", 
                list, lambda x: len(x) > 0, "Please enter at least one symbol."
            )

        if not self.preferred_timeframes:
            self.preferred_timeframes = self._get_validated_input(
                "[Bot]: What timeframes do you prefer? (e.g., M15, H1): ", 
                list, lambda x: len(x) > 0, "Please enter at least one timeframe."
            )

        if self.target_profit <= 0:
            self.target_profit = self._get_validated_input(
                "[Bot]: What's your target profit per trade in dollars?: ", 
                float, lambda x: x > 0, "Please enter a positive amount."
            )

        if self.stop_loss <= 0:
            self.stop_loss = self._get_validated_input(
                "[Bot]: What's your maximum loss per trade (stop loss) in dollars?: ", 
                float, lambda x: x > 0, "Please enter a positive amount."
            )

        if self.risk_percentage <= 0:
            self.risk_percentage = self._get_validated_input(
                "[Bot]: What's your risk percentage per trade? (e.g., 1 for 1%): ", 
                float, lambda x: 0 < x <= 10, "Please enter a percentage between 0.1 and 10."
            )

        if self.max_daily_loss <= 0:
            self.max_daily_loss = self._get_validated_input(
                "[Bot]: What's your maximum daily loss limit in dollars?: ", 
                float, lambda x: x > 0, "Please enter a positive amount."
            )

        # Save and confirm
        self._save_trading_config()

        print(f"\n[Bot]: ✅ Setup complete! Here's your final configuration:")
        print(f"   📊 Symbols: {', '.join(self.trading_symbols)}")
        print(f"   ⏰ Timeframes: {', '.join(self.preferred_timeframes)}")
        print(f"   🎯 Target Profit: ${self.target_profit}")
        print(f"   🛑 Stop Loss: ${self.stop_loss}")
        print(f"   📈 Risk %: {self.risk_percentage}%")
        print(f"   💰 Max Daily Loss: ${self.max_daily_loss}")
        print("\n[Bot]: You can now start trading! Type your questions or commands.")
        print("="*50)
        
    def _save_trading_config(self):
        """Save trading configuration to profile.json file, merging with existing data."""
        config = self._read_json(self.PROFILE_FILE)
        config.update({
            "trading_symbols": self.trading_symbols,
            "target_profit": self.target_profit,
            "stop_loss": self.stop_loss,
            "risk_percentage": self.risk_percentage,
            "max_daily_loss": self.max_daily_loss,
            "preferred_timeframes": self.preferred_timeframes
        })
        self._write_json(self.PROFILE_FILE, config)

    def _load_trading_config(self) -> bool:
        """Load trading configuration from profile.json file."""
        config = self._read_json(self.PROFILE_FILE)
        if "trading_symbols" in config:
            self.trading_symbols = config.get("trading_symbols", [])
            self.target_profit = config.get("target_profit", 0.0)
            self.stop_loss = config.get("stop_loss", 0.0)
            self.risk_percentage = config.get("risk_percentage", 0.0)
            self.max_daily_loss = config.get("max_daily_loss", 0.0)
            self.preferred_timeframes = config.get("preferred_timeframes", [])
            return True
        return False

    # --- Data Formatters ---

    def _format_account_data(self):
        try:
            account = self.broker.getAccountInfo()
            if account is None:
                return {"balance": "N/A", "equity": "N/A", "free_margin": "N/A", "used_margin": "N/A", "margin_level": "N/A"}
            
            return {
                "balance": f"${account.balance:,.2f}" if account.balance else "N/A",
                "equity": f"${account.equity:,.2f}" if account.equity else "N/A",
                "free_margin": f"${account.margin_free:,.2f}" if account.margin_free else "N/A",
                "used_margin": f"${account.margin:,.2f}" if account.margin else "N/A",
                "margin_level": f"{account.margin_level:.2f}" if account.margin_level else "N/A"
            }
        except Exception as e:
            print(f"[Bot][ERROR] Failed to fetch account info: {e}")
            return {"balance": "Error", "equity": "Error", "free_margin": "Error", "used_margin": "Error", "margin_level": "Error"}

    def _format_positions_data(self):
        try:
            positions = self.broker.getPositions()
            if not positions:
                return {"position_count": 0, "positions": "No open positions", "total_volume": 0}
            
            position_list = []
            total_volume = 0
            
            for pos in positions:
                symbol = getattr(pos, 'symbol', "Unknown")
                volume = getattr(pos, 'volume', 0)
                price = getattr(pos, 'price_open', 0)
                type_str = "BUY" if pos.type == 0 else "SELL" if hasattr(pos, 'type') else "Unknown"
                
                position_list.append(f"{symbol} ({type_str} {volume} units @ {price:.5f})")
                total_volume += volume
            
            return {
                "position_count": len(positions),
                "positions": " | ".join(position_list) if position_list else "No positions",
                "total_volume": total_volume
            }
        except Exception as e:
            print(f"[Bot][ERROR] Failed to fetch positions: {e}")
            return {"position_count": 0, "positions": "Error fetching positions", "total_volume": 0}

    def _format_symbols_data(self):
        try:
            if self.trading_symbols:
                symbols_str = ", ".join(self.trading_symbols)
            else:
                symbols = self.broker.getSymbols()
                symbols_str = ", ".join([s.name for s in symbols[:10]]) if symbols else "No symbols available"
            
            return {"symbols": symbols_str}
        except Exception as e:
            print(f"[Bot][ERROR] Failed to fetch symbols: {e}")
            return {"symbols": "Error fetching symbols"}

    def _format_settings_data(self):
        try:
            return {
                "symbols": ", ".join(self.trading_symbols) if self.trading_symbols else "Not configured",
                "target_profit": f"${self.target_profit:,.2f}" if self.target_profit > 0 else "Not set",
                "stop_loss": f"${self.stop_loss:,.2f}" if self.stop_loss > 0 else "Not set",
                "risk_percentage": f"{self.risk_percentage:.1f}" if self.risk_percentage > 0 else "Not set",
                "max_daily_loss": f"${self.max_daily_loss:,.2f}" if self.max_daily_loss > 0 else "Not set",
                "timeframes": ", ".join(self.preferred_timeframes) if self.preferred_timeframes else "Not configured"
            }
        except Exception as e:
            print(f"[Bot][ERROR] Failed to format settings: {e}")
            return {k: "Error" for k in ["symbols", "target_profit", "stop_loss", "risk_percentage", "max_daily_loss", "timeframes"]}

    # --- Interaction & Logging ---

    def _log_interaction(self, user_input: str, intent: str, status: str = "completed"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "intent": intent,
            "status": status
        }
        history = self._read_json(self.HISTORY_FILE, [])
        history.append(entry)
        self._write_json(self.HISTORY_FILE, history[-200:])

    def _get_personalized_greeting(self) -> str:
        stats = self._read_json(self.STATS_FILE)
        counts = stats.get("intent_counts", {})
        if not counts:
            return "Welcome back! How can I help with your trading today?"
        
        favorite = max(counts, key=counts.get)
        greetings = {
            "account_summary": "Welcome back! Want the latest numbers on your equity?",
            "trading_symbols": "Ready to scan the markets? Your usual symbols are ready.",
            "active_positions": "Welcome back. Should we check how your open trades are doing?"
        }
        return greetings.get(favorite, "Welcome back! What's the plan today?")

    def _execute_bulk_scan(self) -> str:
        print("[Bot]: Scanning all symbols in your portfolio... 🔎")
        
        # The Portfolio Manager handles everything now
        results = self.portfolio_manager.evaluate_portfolio_opportunities(
            symbols=self.trading_symbols,
            risk_pct=self.risk_percentage,
            stop_loss=20.0, # Or self.stop_loss
            max_daily_loss=self.max_daily_loss
        )
        
        if not results or all("Portfolio Halt" in r for r in results) and len(results) == 1:
            return results[0] if results else "No high-probability entries right now."
            
        health_report = self.portfolio_manager.get_portfolio_health()
        return "Scan complete:\n" + "\n".join(results) + f"\n\n📊 Status: {health_report}"
    
    def _get_proactive_insight(self) -> tuple[Optional[str], Optional[str]]:
        try:
            account = self.broker.getAccountInfo()
            if not account:
                return None, None
            
            if account.balance > 0:
                equity_ratio = account.equity / account.balance
                if equity_ratio < 0.95:
                    loss_pct = (1 - equity_ratio) * 100
                    return (f"⚠️ Warning: Your equity is down {loss_pct:.1f}%. You've lost ${account.balance - account.equity:,.2f} so far today.", None)
            
            if self.max_daily_loss > 0 and account.balance > 0:
                daily_loss = account.balance - account.equity
                if daily_loss > self.max_daily_loss * 0.8:
                    remaining = self.max_daily_loss - daily_loss
                    return (f"⚠️ Alert: You have ${remaining:,.2f} of your ${self.max_daily_loss:,.2f} daily loss limit remaining.", None)
            
            positions = self.broker.getPositions()
            if (not positions or len(positions) == 0) and self.trading_symbols:
                if len(self.trading_symbols) > 1:
                    return (f"📈 No active positions. Should I scan your portfolio ({len(self.trading_symbols)} symbols) for entry signals?", "bulk_scan")
                return (f"📈 No active positions. Should I scan {self.trading_symbols[0]} for entry signals?", "trading_symbols")
                
            return None, None
        except Exception as e:
            print(f"[Bot][ERROR] Proactive insight failed: {e}")
            return None, None

    def _suggest_config_improvements(self) -> Optional[str]:
        try:
            history = self._read_json(self.HISTORY_FILE, [])
            if len(history) < 5:
                return None
            
            intent_counts = {}
            for entry in history[-50:]:
                intent = entry.get("intent")
                if intent:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            if intent_counts.get("account_summary", 0) >= 5:
                return "💡 I notice you check your balance frequently. Would you like me to send proactive alerts instead? (e.g., 'Alert me if drawdown exceeds 10%')"
            
            if intent_counts.get("active_positions", 0) >= 3 and not self.broker.getPositions():
                return "💡 You've checked positions several times today. Would you like me to auto-notify when conditions are right for trading?"
            
            return None
        except Exception as e:
            print(f"[Bot][ERROR] Config suggestion failed: {e}")
            return None

    def _handle_follow_up_context(self, inp: str) -> Optional[Any]:
        inp_lower = inp.lower()
        if self.last_intent == "account_summary":
            if any(word in inp_lower for word in ["position", "trade", "open", "exposure"]):
                return self.action_mappings["active_positions"](inp)
            elif any(word in inp_lower for word in ["setting", "config", "prefer", "parameter"]):
                return self._format_settings_data()
                
        elif self.last_intent == "active_positions":
            if any(word in inp_lower for word in ["balance", "equity", "margin", "account"]):
                return self.action_mappings["account_summary"](inp)
            elif any(word in inp_lower for word in ["symbol", "pair", "trade"]):
                return self._format_symbols_data()
                
        elif self.last_intent == "trading_symbols":
            if any(word in inp_lower for word in ["setting", "config", "prefer"]):
                return self._format_settings_data()
        
        return None

    def _extract_symbol_from_input(self, inp: str) -> Optional[str]:
        tokens = re.findall(r"[A-Z]{3,6}", inp.upper())
        if not tokens:
            return None
        if self.trading_symbols:
            configured_symbols = {s.upper(): s for s in self.trading_symbols}
            for token in tokens:
                if token in configured_symbols:
                    return configured_symbols[token]
        return tokens[0]

    def _increment_stats(self, tag: str, inp: str):
        stats = self._read_json(self.STATS_FILE, {"intent_counts": {}, "symbol_counts": {}, "watchlist_suggested": {}})
        stats["intent_counts"][tag] = stats["intent_counts"].get(tag, 0) + 1
        symbol = self._extract_symbol_from_input(inp)
        if symbol:
            stats["symbol_counts"][symbol] = stats["symbol_counts"].get(symbol, 0) + 1
        self._write_json(self.STATS_FILE, stats)

    def _suggest_watchlist_adjustment(self) -> Optional[str]:
        stats = self._read_json(self.STATS_FILE, {})
        if sum(stats.get("intent_counts", {}).values()) < 10:
            return None
            
        symbol_counts = stats.get("symbol_counts", {})
        if not symbol_counts:
            return None
            
        top_symbol = max(symbol_counts, key=symbol_counts.get)
        if top_symbol in [s.upper() for s in self.trading_symbols] or stats.get("watchlist_suggested", {}).get(top_symbol):
            return None
            
        stats.setdefault("watchlist_suggested", {})[top_symbol] = True
        self._write_json(self.STATS_FILE, stats)
        return f"💡 I notice you check {top_symbol} frequently. Should I add it to your permanent watch list?"

    def _execute_action(self, tag: str, inp: str):
        intent_data = next((tg for tg in self.data["intents"] if tg['tag'] == tag), None)
        if tag in self.action_mappings:
            live_data = self.action_mappings[tag](inp)
            response_template = random.choice(intent_data['responses']) if intent_data else "[Bot]: Unable to process request"
            try:
                if isinstance(live_data, dict):
                    print("[Bot]:", response_template.format(**live_data))
                else:
                    print("[Bot]:", response_template.format(data=live_data))
            except KeyError:
                print(f"[Bot]: {response_template} (Additional Info: {live_data})")
        elif intent_data:
            print("[Bot]:", random.choice(intent_data['responses']))
        else:
            print("[Bot]: I couldn't find a response for that.")

        self.last_intent = tag
        self.active_suggestion = None
        self._log_interaction(inp, tag, "completed")
        self._increment_stats(tag, inp)
        watchlist_suggestion = self._suggest_watchlist_adjustment()
        if watchlist_suggestion:
            print(f"[Bot]: {watchlist_suggestion}")

    def _handle_intent(self, inp: str) -> bool:
        clean_inp = inp.lower().strip()
        rejection_words = ["nah", "no", "stop", "don't", "dont", "nope", "not now"]
        confirmation_words = ["yes", "yeah", "yep", "sure", "ok", "please", "do it", "go ahead", "yup"]
        positive_feedback = ["good", "wow", "nice", "awesome", "great", "thanks", "thank you"]
        
        for word in rejection_words:
            if clean_inp.startswith(word):
                print("[Bot]: No problem, I won't do that for now.")
                self.active_suggestion = None
                inp = re.sub(rf"^{word}[,\s]*", "", inp, flags=re.IGNORECASE)
                break

        if any(word in clean_inp for word in ["portfolio", "watchlist", "add symbol"]):
            return self._handle_portfolio_update(inp)

        bag = self._bag_of_words(inp)
        results = self.model.predict(bag, verbose=0)[0]
        results_index = np.argmax(results)
        tag = self.labels[results_index]
        confidence = results[results_index]

        if any(word in clean_inp for word in ["symbol", "pair", "trade", "instrument"]):
            self._execute_action("trading_symbols", inp)
            return True

        if self.active_suggestion and any(word in clean_inp for word in confirmation_words) and not any(word in clean_inp for word in rejection_words):
            self._execute_action(self.active_suggestion, inp)
            return True
            
        if any(word in clean_inp for word in positive_feedback) and confidence < 0.5:
            self._execute_action("gratitude", inp)
            return True
            
        if any(word in clean_inp for word in ["status", "report", "summary"]):
            self._execute_action("account_summary", inp)
            self._execute_action("trading_settings", inp)
            return True

        if confidence > 0.7:
            self._execute_action(tag, inp)
            return True

        follow_up_data = self._handle_follow_up_context(inp)
        if follow_up_data is not None:
            intent_data = next((tg for tg in self.data["intents"] if tg['tag'] == self.last_intent), None)
            response_template = random.choice(intent_data['responses']) if intent_data else "Got it."
            try:
                if isinstance(follow_up_data, dict):
                    print("[Bot]:", response_template.format(**follow_up_data))
                else:
                    print("[Bot]:", response_template.format(data=follow_up_data))
            except Exception:
                print("[Bot]: Based on your last question:", follow_up_data)
            self._log_interaction(inp, f"{self.last_intent}_followup", "completed")
            return True

        print("[Bot]: I didn't quite get that. Could you try rephrasing?")
        self._log_interaction(inp, "unrecognized", "failed")
        return False

    def _handle_portfolio_update(self, inp: str) -> bool:
        try:
            broker_symbols = self.broker.getSymbols()
            if not broker_symbols:
                print("[Bot]: ⚠️ I couldn't fetch symbols from the broker. Please check your connection.")
                return False
                
            all_available_names = [s.name.upper() for s in broker_symbols]
            inp_lower = inp.lower()
            found_symbols = []

            if "all" in inp_lower:
                keywords = ["USD", "GBP", "EUR", "JPY", "AUD", "CAD", "CHF", "NZD", "XAU", "BTC"]
                target_currency = next((k for k in keywords if k.lower() in inp_lower), None)
                
                if target_currency:
                    found_symbols = [s for s in all_available_names if target_currency in s]
                    print(f"[Bot]: Searching for all {target_currency} pairs...")
                else:
                    print("[Bot]: Which group of symbols should I add? (e.g., 'all USD')")
                    return False
            else:
                extracted = self._extract_symbol_from_input(inp)
                if extracted and extracted.upper() in all_available_names:
                    found_symbols = [extracted.upper()]
                elif extracted:
                    print(f"[Bot]: ❌ {extracted} is not available on your MT5 account.")
                    return False

            if found_symbols:
                self.trading_symbols = list(set(self.trading_symbols + found_symbols))
                self._save_trading_config()
                
                print(f"[Bot]: ✅ Success! Added: {', '.join(found_symbols)}")
                print(f"   📊 Current Portfolio: {', '.join(self.trading_symbols)}")
                return True
                
        except Exception as e:
            print(f"[Bot][ERROR]: Portfolio update failed: {e}")
        
        return False
    
    
    def _parse_conversational_config(self, text: str) -> dict:
        """Extracts trading configurations from a natural language sentence."""
        text_upper = text.upper()
        
        # 1. Extract Symbols (e.g., EURUSD, BTCUSD)
        symbols = list(set(re.findall(r'\b[A-Z]{6}\b', text_upper)))
        
        # 2. Extract Timeframes (e.g., H1, M15)
        timeframes = list(set(re.findall(r'\b(M1|M5|M15|M30|H1|H4|D1|W1|MN)\b', text_upper)))
        
        # 3. Extract Risk Percentage (e.g., "1%", "2.5 %")
        risk_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        risk = float(risk_match.group(1)) if risk_match else 0.0
        
        # 4. Extract Dollar Amounts based on context words
        target_profit = 0.0
        stop_loss = 0.0
        max_daily_loss = 0.0
        
        # Look for profit keywords (e.g., "make $10", "profit of 20", "target $5")
        profit_match = re.search(r'(?i)(?:make|profit|target|gain).*?\$?(\d+(?:\.\d+)?)', text)
        if profit_match: 
            target_profit = float(profit_match.group(1))
            
        # Look for daily loss keywords (e.g., "max daily loss $20", "lose $50 daily")
        daily_loss_match = re.search(r'(?i)(?:daily|max.*?loss).*?\$?(\d+(?:\.\d+)?)', text)
        if daily_loss_match: 
            max_daily_loss = float(daily_loss_match.group(1))
            
        # Look for stop loss keywords (e.g., "stop loss $5", "sl 10")
        sl_match = re.search(r'(?i)(?:stop\s*loss|sl).*?\$?(\d+(?:\.\d+)?)', text)
        if sl_match: 
            stop_loss = float(sl_match.group(1))
        
        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "risk": risk,
            "target_profit": target_profit,
            "stop_loss": stop_loss,
            "max_daily_loss": max_daily_loss
        }
    # --- ML / Model Processing ---

    def _load_intents(self):
        with self.intents_filepath.open("r") as file:
            self.data = json.load(file)

    def _process_data(self):
        if self.DATA_PICKLE.exists():
            with self.DATA_PICKLE.open("rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            docs_x = []
            docs_y = []

            for intent in self.data['intents']:
                for pattern in intent['patterns']:
                    wrds = nltk.word_tokenize(pattern)
                    self.words.extend(wrds)
                    docs_x.append(wrds) 
                    docs_y.append(intent['tag'])
                    
                if intent['tag'] not in self.labels:
                    self.labels.append(intent['tag'])

            self.words = sorted(list(set([self.stemmer.stem(w.lower()) for w in self.words if w not in ["?", "!", ".", ","]])))
            self.labels = sorted(self.labels)

            out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(docs_x):
                bag = []
                wrds = [self.stemmer.stem(w.lower()) for w in doc]
                for w in self.words:
                    bag.append(1 if w in wrds else 0)
                    
                output_row = out_empty[:]
                output_row[self.labels.index(docs_y[x])] = 1
                
                self.training.append(bag)
                self.output.append(output_row)

            self.training = np.array(self.training)
            self.output = np.array(self.output)

            with self.DATA_PICKLE.open("wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

    def _build_model(self):
        if self.MODEL_FILE.exists():
            self.model = load_model(str(self.MODEL_FILE))
        else:
            self.model = Sequential([
                Dense(8, input_shape=(len(self.training[0]),), activation='relu'),
                Dense(8, activation='relu'),
                Dense(len(self.output[0]), activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(self.training, self.output, epochs=200, batch_size=8, verbose=1)
            self.model.save(str(self.MODEL_FILE))

    def _bag_of_words(self, s: str) -> np.ndarray:
        bag = [0 for _ in range(len(self.words))]
        s_words = [self.stemmer.stem(word.lower()) for word in nltk.word_tokenize(s)]
        
        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    bag[i] = 1
                    
        return np.array([bag])

    # --- Main Loop ---

    def start_chat(self):
        print("")
        
        data = self._read_json(self.PROFILE_FILE)
        login = data.get("login")
        password = data.get("password")
        server = data.get("server")
        
        if (login and password and server) or not self.broker.connected:
            print(f"[Bot][SYSTEM]: Found saved profile for {login}. Verifying...")
            login_res = self.broker.connect(login=login, password=password, server=server)
            if self.broker.connected:
                print(f"[Bot]: Welcome back! Your account is ready.")
            else:
                print("[Bot]: [API] Saved profile could not be verified. Please log in again.")

        while not self.broker.connected:
            try:
                # Handled to prevent crash if user inputs letters instead of integers
                login_input = int(input("[Bot]: Enter your MT5 Account Number: ").strip())
            except ValueError:
                print("[Bot]: Account Number must be a valid integer. Try again.")
                continue
                
            if len(str(login_input)) < 5:
                print("[Bot]: Check your login and try again.")
                continue
            
            # NOTE: Plain text password storage is discouraged in production. 
            # Consider utilizing python's `keyring` library or encrypting the json values.
            password_input = getpass.getpass("[Bot]: Enter your MT5 Password: ")
            if not password_input:
                print("[Bot]: Check your password and try again.")
                continue

            server_input = input("[Bot]: Enter your Broker Server (e.g., MetaQuotes-Demo): ").strip()
            self.broker.connect(login=login_input, password=password_input, server=server_input)
           
            if self.broker.connected:
                data.update({
                    "login": login_input,
                    "password": password_input,
                    "server": server_input
                })
                self._write_json(self.PROFILE_FILE, data)
                print(f"[Bot]: Account linked! Welcome, #{self.broker.login}.")
            else:
                print("[Bot]: Failed to connect to broker. Please verify your credentials.")
        
        if not self._load_trading_config():
            self.setup_trading_config()
        else:
            greeting = self._get_personalized_greeting()
            print(f"\n[Bot]: {greeting}")
            print(f"   📊 Watchlist: {', '.join(self.trading_symbols)}")
            print(f"   🎯 Risk: {self.risk_percentage}%")
            
            update_config = input("\n[Bot]: Would you like to update your trading configuration? (y/n): ").strip().lower()
            if update_config in ['y', 'yes']:
                self.setup_trading_config()
        
        print("\n--- Bot is ready! Start talking (type 'quit' to stop, 'switch' to change user) ---")
        try:
            interaction_count = 0
            while True:
                inp = input("You: ").strip()
                if not inp: continue
                
                if inp.lower() == "quit":
                    break
                elif inp.lower() == "switch":
                    self.user_anchor = None
                    if self.PROFILE_FILE.exists():
                        self.PROFILE_FILE.unlink() 
                    print("[Bot]: Switching user...")
                    self.start_chat()  
                    return
                
                if self._handle_intent(inp):
                    insight, suggestion_tag = self._get_proactive_insight()
                    if insight:
                        print("[Bot]:", insight)
                        self.active_suggestion = suggestion_tag
                
                # Periodically suggest config improvements (every 20 interactions)
                interaction_count += 1
                if interaction_count % 20 == 0:
                    suggestion = self._suggest_config_improvements()
                    if suggestion:
                        print(f"\n[Bot]: {suggestion}\n")
                        
        except KeyboardInterrupt:
            print("\n[Bot]: Chat session ended. Goodbye!")
            return