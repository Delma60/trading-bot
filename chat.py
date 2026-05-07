
import json
import random
import re
import getpass
import threading
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional
from prompt_toolkit.patch_stdout import patch_stdout

from manager.gemini_engine import GeminiEngine

from rich.console import Console
from prompt_toolkit import PromptSession
from manager.nlp_engine import NLPEngine
from manager.portfolio_manager import PortfolioManager
from manager.risk_manager import RiskManager
from manager.profile_manager import ProfileManager
from strategies.strategy_manager import StrategyManager
from trader import Trader

class Chatbot(ProfileManager, NLPEngine, GeminiEngine):
    """Handles NLP, Model Training, and the conversational trading interface."""
    
    # Centralized file paths
    DATA_DIR = Path("data")
    PROFILE_FILE = DATA_DIR / "profile.json"
    HISTORY_FILE = DATA_DIR / "history.json"
    STATS_FILE = DATA_DIR / "stats.json"
    DATA_PICKLE = DATA_DIR / "data.pickle"
    MODEL_FILE = DATA_DIR / "chatbot_model.keras"
    SYMBOLS_CACHE_DIR = DATA_DIR / "symbols"

    def __init__(self, intents_filepath: str, broker: Trader, strategy_manager: StrategyManager, portfolio_manager: PortfolioManager, risk_manager: RiskManager):
        ProfileManager.__init__(self, data_dir=str(self.DATA_DIR))
        NLPEngine.__init__(self, intents_filepath=intents_filepath, data_dir=str(self.DATA_DIR))
        GeminiEngine.__init__(self)
        self.stemmer = LancasterStemmer()
        self.intents_filepath = Path(intents_filepath)
        self.broker = broker  
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager

        self.notification_inbox = []
        self.inbox_lock = threading.Lock()

        self.last_intent = None  
        self.active_suggestion = None  

        # Trading configuration state
        self.trading_symbols = []
        self.daily_goal = 0.0
        self.target_profit = 0.0
        self.stop_loss = 0.0
        self.risk_percentage = 0.0
        self.max_daily_loss = 0.0
        self.preferred_timeframes = []
        self.symbol_cache = []
        self.symbol_lookup = {}

        # === CONTEXTUAL MEMORY: Remember trading context across turns ===
        self.memory = {
            "last_symbol": None,
            "last_timeframe": "H1",
            "last_money_amount": None
        }

        # === MULTI-TURN DIALOGUE: Track incomplete command state ===
        self.pending_action = None  # e.g., "awaiting_trade_symbol", "awaiting_amount"
        self.pending_data = {}      # Store data collected so far in the incomplete command

        # Action mappings
        self.action_mappings = {
            "active_positions": lambda _: self._format_positions_data(),
            "account_summary": lambda _: self._format_account_data(),
            "trading_symbols": lambda _: self._format_symbols_data(),
            "trading_settings": lambda _: self._format_settings_data(),
            "update_config": lambda _: self.setup_trading_config(),
            "bulk_scan": lambda _: self._run_autonomous_scan(),
            "check_notifications": lambda _: self._read_inbox(),
            "close_all": lambda _: self.broker.close_all_positions(), # <-- ADD THIS LINE
            "retrain_model": lambda _: self._retrain_model(),
        }

        
        self.console = Console()
        self.session = PromptSession()

        self._load_local_symbols()

    # --- IO Helpers ---
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

    def _read_json(self, filepath: Path, default: Any = None) -> Any:
        if not filepath.exists(): return default if default is not None else {}
        try:
            with filepath.open("r") as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return default if default is not None else {}

    def receive_system_alert(self, msg: str, priority: str = "normal"):
        """
        Background engines call this to store notifications.
        Only interrupts the console for critical or trade events.
        """
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        # Filter out 'normal' noise to clean up the terminal
        if priority in ["critical", "trade_executed"]:
            from prompt_toolkit.patch_stdout import patch_stdout
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            with patch_stdout():
                if priority == "critical":
                    print(f"\n[{timestamp}] [Bot]: 🚨 CRITICAL: {msg}")
                elif priority == "trade_executed":
                    print(f"\n[{timestamp}] [Bot]: ✅ TRADE: {msg}")
            

    def bot_print(self, msg: str):
        """Centralized printer to enforce the [time] [Bot]: format."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Strip out any existing '[Bot]:' text so it doesn't double up
        clean_msg = msg.replace("[Bot]: ", "").replace("[Bot]:", "").strip()
        print(f"[{timestamp}] [Bot]: {clean_msg}")

    def _read_inbox(self):
        with self.inbox_lock:
            if not self.notification_inbox:
                return "Your inbox is empty. The scanner hasn't reported anything new."

            items = list(self.notification_inbox)
            self.notification_inbox.clear()

        lines = [f"You have {len(items)} unread system events:"]
        for item in items:
            prefix = "🟢" if item['priority'] == "trade_executed" else "ℹ️"
            if item['priority'] == "critical":
                prefix = "⚠️"
            lines.append(f"  {prefix} {item['msg']}")

        return "\n".join(lines)

    def _write_json(self, filepath: Path, data: Any):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as f: json.dump(data, f, indent=4)

    def _get_validated_input(self, prompt: str, cast_type: type, validation_func: Callable[[Any], bool], error_msg: str) -> Any:
        while True:
            raw_input = input(prompt).strip()
            if not raw_input: continue
            try:
                val = [i.strip().upper() for i in raw_input.split(',')] if cast_type == list else cast_type(raw_input)
                if validation_func(val): return val
                self.bot_print(error_msg)
            except ValueError: self.bot_print(f"Please enter a valid {cast_type.__name__}.")

    def _symbol_file_path(self, symbol: str) -> Path:
        return self.SYMBOLS_CACHE_DIR / f"{symbol.upper()}.json"

    def _load_local_symbols(self) -> list:
        symbols = []
        if not self.SYMBOLS_CACHE_DIR.exists():
            return []

        for file in self.SYMBOLS_CACHE_DIR.glob("*.json"):
            if file.is_file():
                try:
                    data = self._read_json(file, {})
                    if isinstance(data, dict) and data.get("symbol"):
                        symbols.append(data)
                except Exception:
                    continue

        self.symbol_cache = symbols
        self.symbol_lookup = {item["symbol"].upper(): item for item in symbols}
        return symbols

    def _save_local_symbol(self, symbol_data: dict):
        self.SYMBOLS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        symbol_file = self._symbol_file_path(symbol_data.get("symbol", "UNKNOWN"))
        self._write_json(symbol_file, symbol_data)
        self.symbol_cache = [item for item in self.symbol_cache if item.get("symbol", "").upper() != symbol_data.get("symbol", "").upper()]
        self.symbol_cache.append(symbol_data)
        self.symbol_lookup[symbol_data.get("symbol", "").upper()] = symbol_data

    def _refresh_symbol_cache(self) -> list:
        symbols = self.broker.getSymbols()
        if not symbols:
            return self._load_local_symbols()

        symbol_data = []
        for symbol in symbols:
            symbol_name = getattr(symbol, 'name', None) or getattr(symbol, 'symbol', None)
            if not symbol_name:
                continue
            
            symbol_record = {
                "symbol": symbol_name,
                "description": getattr(symbol, 'description', ''),
                "bid": getattr(symbol, 'bid', 0),
                "ask": getattr(symbol, 'ask', 0)
            }
            symbol_data.append(symbol_record)

        self.symbol_cache = symbol_data
        self.symbol_lookup = {item['symbol'].upper(): item for item in symbol_data}
        return symbol_data

    def _get_symbol_cache(self) -> list:
        if self.symbol_cache:
            return self.symbol_cache
        cached = self._load_local_symbols()
        if cached:
            return cached
        return self._refresh_symbol_cache()

    def _infer_asset_class(self, symbol: str) -> str:
        symbol = symbol.upper()
        if any(symbol.startswith(prefix) for prefix in ["XAU", "XAG", "XPT", "XPD"]):
            return "Metals"
        if any(token in symbol for token in ["BTC", "ETH", "LTC", "XBT", "USDT", "DOGE"]):
            return "Crypto"
        if any(token in symbol for token in ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]):
            return "Forex"
        return "Forex"

    def _find_symbols_in_text(self, inp: str) -> list:
        candidates = re.findall(r"\b[A-Z]{6,7}\b", inp.upper())
        valid_symbols = []
        cache = self._get_symbol_cache()
        available = {item['symbol'].upper() for item in cache if isinstance(item, dict) and item.get('symbol')}

        for symbol in candidates:
            if symbol in available or len(symbol) == 6:
                valid_symbols.append(symbol)
        return list(dict.fromkeys(valid_symbols))

    def _add_symbols_to_portfolio(self, symbols: list) -> list:
        if not symbols:
            return []

        added = []
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol not in self.trading_symbols:
                self.trading_symbols.append(symbol)
                added.append(symbol)

            if hasattr(self.portfolio_manager, 'add_symbol'):
                self.portfolio_manager.add_symbol(symbol, asset_class=self._infer_asset_class(symbol))

        if added:
            self._save_trading_config()
            self._update_max_open_trades()
        return added

    def _get_portfolio_symbols(self) -> list:
        if self.trading_symbols:
            return self.trading_symbols
        self._load_trading_config()
        return self.trading_symbols

    def _update_max_open_trades(self):
        """Set max exposure equal to the number of tracked trading symbols."""
        self.risk_manager.max_open_trades = max(1, len(self.trading_symbols))
        # Silenced print to keep startup clean

    def _parse_timeframe_from_text(self, text: str) -> str:
        match = re.search(r'\b(M1|M5|M15|M30|H1|H4|D1)\b', text.upper())
        return match.group(1) if match else "H1"

    def _normalize_ohlcv_dataframe(self, df):
        if df is None or df.empty:
            return df

        if 'time' in df.columns:
            try:
                df = df.copy()
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df = df.drop(columns=['time'], errors='ignore')
            except Exception:
                pass

        volume_col = None
        for candidate in ['volume', 'tick_volume', 'real_volume']:
            if candidate in df.columns:
                volume_col = candidate
                break

        keep_cols = ['datetime', 'open', 'high', 'low', 'close']
        if volume_col:
            keep_cols.append(volume_col)
            df = df[keep_cols]
            df = df.rename(columns={volume_col: 'volume'})
        else:
            df = df[[col for col in keep_cols if col in df.columns]]

        return df

    def _download_portfolio_ohlcv(self, timeframe: str = "H1", count: int = 200):
        symbols = self._get_portfolio_symbols()
        if not symbols:
            self.bot_print("No portfolio symbols found in profile.json. Add symbols to your portfolio first.")
            return

        if not self.broker.connected:
            self.bot_print("I need MT5 connected to download OHLCV data.")
            return

        output_dir = self.DATA_DIR / "ohlcv"
        output_dir.mkdir(parents=True, exist_ok=True)

        for symbol in symbols:
            self.bot_print(f"Downloading OHLCV for {symbol} ({timeframe})...")
            df = self.broker.get_historical_rates(symbol, timeframe=timeframe, count=count)
            if df is None or df.empty:
                self.bot_print(f"Failed to fetch OHLCV for {symbol}.")
                continue

            df = self._normalize_ohlcv_dataframe(df)
            if df is None or df.empty:
                self.bot_print(f"No valid OHLCV data for {symbol}.")
                continue

            csv_path = output_dir / f"{symbol}.csv"
            try:
                df.to_csv(csv_path, index=False)
                self.bot_print(f"Saved {symbol} OHLCV to {csv_path}")
            except Exception as e:
                self.bot_print(f"Failed to write CSV for {symbol}: {e}")

        self.bot_print("Portfolio OHLCV download complete.")

    def _handle_symbol_command(self, inp: str) -> bool:
        clean_inp = inp.lower().strip()
        if (
            any(keyword in clean_inp for keyword in ["generate ohlc", "generate ohlcv", "generate ohlvc", "download ohlc", "download ohlcv", "export ohlc", "export ohlcv", "download csv", "export csv", "ohlc csv", "ohlcv csv"]) or
            ("generate" in clean_inp and "portfolio" in clean_inp) or
            ("download" in clean_inp and "portfolio" in clean_inp)
        ):
            timeframe = self._parse_timeframe_from_text(inp)
            self._download_portfolio_ohlcv(timeframe=timeframe)
            return True

        if any(keyword in clean_inp for keyword in ["available symbols", "show symbols", "list symbols", "my symbols", "what symbols", "portfolio symbols"]):
            self._show_available_symbols()
            return True

        if any(keyword in clean_inp for keyword in ["add to portfolio", "add to watchlist", "add symbol", "add symbols", "track", "watchlist"]) or (
            "add" in clean_inp and "portfolio" in clean_inp
        ):
            symbols = self._find_symbols_in_text(inp)
            if not symbols:
                self.bot_print("I couldn't determine which symbol to add. Try 'Add AUDCAD to portfolio'.")
                return True

            added = self._add_symbols_to_portfolio(symbols)
            if added:
                self.bot_print(f"✅ Added {', '.join(added)} to your watchlist.")
            else:
                self.bot_print(f"Those symbols are already in your list: {', '.join(symbols)}")
            return True

        return False

    def _show_available_symbols(self):
        symbols = self._get_portfolio_symbols()
        if not symbols:
            self.bot_print("No portfolio symbols are configured. Add some symbols to profile.json first.")
            return

        self.bot_print("Your portfolio symbols from profile.json:")
        for idx, symbol in enumerate(symbols, start=1):
            print(f"  {idx}. {symbol}")
        self.bot_print("Use 'Download OHLCV' to save historical data for these portfolio symbols.")

    # --- Conversational Parser ---

    def _parse_conversational_config(self, text: str) -> dict:
        """Extracts trading parameters from natural language[cite: 6]."""
        text_upper = text.upper()
        symbols = list(set(re.findall(r'\b[A-Z]{6}\b', text_upper)))
        timeframes = list(set(re.findall(r'\b(M1|M5|M15|M30|H1|H4|D1)\b', text_upper)))
        
        risk_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        daily_profit_match = re.search(r'(?i)(?:make|profit|target).*?\$?(\d+(?:\.\d+)?)\s*(?:daily|per day)', text)
        
        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "risk": float(risk_match.group(1)) if risk_match else 0.0,
            "daily_profit": float(daily_profit_match.group(1)) if daily_profit_match else 0.0
        }

    # --- Trading Setup ---

    def setup_trading_config(self):
        """Conversational setup for trading configuration[cite: 6]."""
        print("\n" + "="*30 + "\n🤖 TRADING SETUP\n" + "="*30)
        user_input = input("[Bot]: Describe your goal (e.g., 'Risk 1%, make $10 daily on EURUSD'): ").strip()
        
        parsed = self._parse_conversational_config(user_input)
        self.trading_symbols = parsed["symbols"]
        self.risk_percentage = parsed["risk"]
        
        if parsed["daily_profit"] > 0:
            self.daily_goal = parsed["daily_profit"]
            
            # Example: We want to make 10% of our daily goal per session
            session_goal = self.daily_goal / 10.0 
            
            # Spread the session goal across your active symbols
            active_symbols_count = len(self.trading_symbols) if self.trading_symbols else 1
            self.target_profit = round(session_goal / active_symbols_count, 2)
            
            self.bot_print(f"Daily Goal set to ${self.daily_goal}.")
            self.bot_print(f"Session target is ${session_goal}. With {active_symbols_count} symbols, target per trade is ${self.target_profit}.")

        # Fill remaining gaps[cite: 6]
        if not self.trading_symbols:
            self.trading_symbols = self._get_validated_input("[Bot]: Symbols (e.g. EURUSD): ", list, lambda x: len(x) > 0, "Enter symbols.")
        if self.risk_percentage <= 0:
            self.risk_percentage = self._get_validated_input("[Bot]: Risk % per trade: ", float, lambda x: 0 < x <= 10, "Enter 0.1 to 10.")
        if self.max_daily_loss <= 0:
            self.max_daily_loss = self._get_validated_input("[Bot]: Max daily loss ($): ", float, lambda x: x > 0, "Enter a positive amount.")

        self._save_trading_config()
        self._update_max_open_trades()
        self.bot_print("✅ Configuration saved.")

    def _save_trading_config(self):
        config = self._read_json(self.PROFILE_FILE)
        config.update({
            "trading_symbols": self.trading_symbols,
            "daily_goal": self.daily_goal,
            "target_profit": self.target_profit,
            "stop_loss": self.stop_loss,
            "risk_percentage": self.risk_percentage,
            "max_daily_loss": self.max_daily_loss,
            "preferred_timeframes": self.preferred_timeframes
        })
        self._write_json(self.PROFILE_FILE, config)

    def _format_symbols_data(self):
        if self.trading_symbols:
            return {"symbols": ", ".join(self.trading_symbols)}

        cache = self._get_symbol_cache()
        if cache:
            return {"symbols": ", ".join([item["symbol"] for item in cache[:20]])}

        return {"symbols": "No symbols available."}

    def _load_trading_config(self) -> bool:
        """Load trading configuration from profile.json file."""
        config = self._read_json(self.PROFILE_FILE)
        if "trading_symbols" in config:
            self.trading_symbols = config.get("trading_symbols", [])
            self.daily_goal = config.get("daily_goal", 0.0)
            self.target_profit = config.get("target_profit", 0.0)
            self.stop_loss = config.get("stop_loss", 0.0)
            self.risk_percentage = config.get("risk_percentage", 0.0)
            self.max_daily_loss = config.get("max_daily_loss", 0.0)
            self.preferred_timeframes = config.get("preferred_timeframes", [])
            self._update_max_open_trades()
            return True
        return False

    # --- Trading Execution ---

    def _run_autonomous_scan(self):
        """Triggers the Portfolio Manager to scan and execute trades[cite: 6]."""
        self.bot_print("Running autonomous portfolio scan... 🔎")
        results = self.portfolio_manager.evaluate_portfolio_opportunities(
            risk_pct=self.risk_percentage,
            stop_loss=20.0, # Default pips
            max_daily_loss=self.max_daily_loss
        )
        for r in results: self.bot_print(r)
        self.bot_print(self.portfolio_manager.get_portfolio_health())

    def _retrain_model(self):
        """Manually trigger continuous learning for the last symbol."""
        symbol = self.memory.get("last_symbol")
        if not symbol:
            self.bot_print("No last symbol in memory. Please mention a symbol first.")
            return
        self.strategy_manager.continuous_learning_routine(symbol)
        self.bot_print(f"✅ Retrained model for {symbol}.")

    

    def _handle_intent(self, inp: str) -> bool:
        """Processes user input with entity extraction, memory, and multi-turn dialogue."""
        clean_inp = inp.lower().strip()
        
        # === STEP 1: If bot is in middle of a multi-turn action, handle completion first ===
        if self.pending_action:
            return self._handle_pending_action(inp)
        
        # === STEP 2: Extract entities from the user's text ===
        entities = self.extract_entities(inp)
        
        # === STEP 3: Update memory with newly mentioned entities ===
        if entities["symbols"]:
            self.memory["last_symbol"] = entities["symbols"][0]
        if entities["timeframes"]:
            self.memory["last_timeframe"] = entities["timeframes"][0]
        if entities["money"]:
            self.memory["last_money_amount"] = entities["money"][0]
        
        if len(clean_inp) <= 2 and clean_inp not in ["hi", "yo"]:
            self.bot_print("👍")
            return True
        
        # === STEP 4: Handle symbol-based commands ===
        if self._handle_symbol_command(inp):
            return True

        # === STEP 4.5: Handle direct commands ===
        if "retrain" in clean_inp and "model" in clean_inp:
            self._execute_action("retrain_model", inp, entities)
            return True

        # === STEP 5: Predict intent via Keras model ===
        if any(word in clean_inp for word in ["start", "scan"]):
            intent_tag = "bulk_scan"
            confidence = 1.0
        else:
            intent_tag, confidence = self.predict_intent(inp)
        
        # === STEP 6: Execute with high confidence, or ask for clarification ===
        if confidence >= 0.75:
            self._execute_action(intent_tag, inp, entities)
            return True
        elif 0.30 <= confidence < 0.75:
            self.bot_print("I'm a bit unsure. Asking my Gemini brain to confirm...")
            
            # 1. Load valid intents from your JSON
            with open(self.intents_filepath, 'r') as f:
                valid_intents = [i['tag'] for i in json.load(f)['intents']]
            
            # chat.py (around line 515)

            gemini_raw = self.route_intent(inp, valid_intents, intent_tag)

            # Case 1: Gemini matched a fixed intent
            if gemini_raw in valid_intents:
                self.bot_print(f"Gemini mapped this to '{gemini_raw}'. Executing!")
                self._execute_action(gemini_raw, inp, entities)
                return True

            # Case 2: Gemini provides a direct answer (General Chat)
            elif gemini_raw.startswith("GENERAL_CHAT:"):
                response = gemini_raw.replace("GENERAL_CHAT:", "").strip()
                self.bot_print(response)
                return True

            # Case 3: Gemini suggests a new intent we don't have yet
            elif gemini_raw.startswith("SUGGEST_NEW:"):
                suggestion = gemini_raw.replace("SUGGEST_NEW:", "").strip()
                new_tag, suggested_response = suggestion.split("|")
                
                self.bot_print(f"I don't have a specific command for that, but I think you mean '{new_tag.strip()}'.")
                self.bot_print(suggested_response.strip())
                
                # Ask if the bot should learn this
                self.bot_print(f"Should I learn this phrase and add it to my '{new_tag.strip()}' module? (y/n)")
                self.pending_action = "confirm_learning"
                self.pending_data = {
                        "tag": new_tag.strip(),
                        "pattern": inp,
                        "response": suggested_response.strip()
                    }
                return True

            else:
                self.bot_print("I'm still learning. Try asking to 'scan' or 'show portfolio'.")
                return False

        # Low confidence: Total failure
        else:
            self.bot_print("I don't understand that yet. What did you mean?")
            self.pending_action = "teach_new_phrase"
            self.pending_data = {"original_input": inp}
            return True
            
    
    def _handle_pending_action(self, inp: str) -> bool:
        """Handles completion of incomplete commands (multi-turn dialogue)."""
        if self.pending_action == "awaiting_trade_symbol":
            entities = self.extract_entities(inp)
            if entities["symbols"]:
                symbol = entities["symbols"][0]
                self.memory["last_symbol"] = symbol
                self.bot_print(f"✅ Executing trade for {symbol}...")
                # Execute the pending trade with the provided symbol
                self.pending_data["symbol"] = symbol
                self._execute_action(self.pending_data["intent"], f"trade {symbol}", entities)
                self.pending_action = None
                self.pending_data = {}
                return True
            elif inp.lower().strip() == "cancel":
                self.bot_print("Trade cancelled.")
                self.pending_action = None
                self.pending_data = {}
                return True
            else:
                self.bot_print("I still didn't catch the symbol. Please reply with something like 'EURUSD', 'GBPUSD', or type 'cancel'.")
                return False
        
        elif self.pending_action == "awaiting_amount":
            entities = self.extract_entities(inp)
            if entities["money"]:
                amount = entities["money"][0]
                self.memory["last_money_amount"] = amount
                self.bot_print(f"✅ Setting amount to ${amount}...")
                self.pending_data["amount"] = amount
                self._execute_action(self.pending_data["intent"], inp, entities)
                self.pending_action = None
                self.pending_data = {}
                return True
            elif inp.lower().strip() == "cancel":
                self.bot_print("Action cancelled.")
                self.pending_action = None
                self.pending_data = {}
                return True
            else:
                self.bot_print("I couldn't parse the amount. Please try again with a number like '100' or '50.5', or type 'cancel'.")
                return False
        
        elif self.pending_action == "confirm_learning":
            tag = self.pending_data["predicted_tag"]
            original_input = self.pending_data["original_input"]
            new_response = self.pending_data.get("response") # The Gemini-generated response
            
            if inp.lower() in ["y", "yes"]:
                # Update intents.json with both the pattern and the NEW response
                self.add_intent_pattern(tag, original_input, notify_callback=self.receive_system_alert)
                # Note: You may need to add a method to NLPEngine to save responses as well!
                self.bot_print("✅ Knowledge updated. I'll know what to do next time!")
                
        elif self.pending_action == "teach_new_phrase":
            # For now, we just reset. (In the future, you can build a menu here to pick the right intent).
            self.bot_print("Let's start over. What would you like to do?")
            self.pending_action = None
            self.pending_data = {}
            return True
        # === SELF-CORRECTION RETRY LOGIC ===
        elif self.pending_action == "retry_micro_lot":
            if inp.lower() in ["y", "yes", "yeah", "sure"]:
                symbol = self.pending_data["symbol"]
                order_type = self.pending_data["order_type"]
                
                self.bot_print(f"Attempting emergency execution with 0.01 lots for {symbol}...")
                result = self.broker.execute_trade(
                    symbol=symbol,
                    order_type=order_type,
                    lots=0.01, # The micro-lot override
                    stop_loss_pips=self.stop_loss,
                    take_profit_pips=self.target_profit
                )
                
                if result and result.get("success"):
                    self.bot_print(f"✅ Success! Micro-lot trade secured. Ticket #{result.get('ticket')}")
                else:
                    self.bot_print(f"❌ Failed again. Broker error: {result.get('comment') if result else 'Unknown'}")
            else:
                self.bot_print("Understood. Canceling trade attempt.")
                
            self.pending_action = None
            self.pending_data = {}
            return True
        
        elif self.pending_action == "awaiting_close_symbol":
            entities = self.extract_entities(inp)
            if entities["symbols"]:
                symbol = entities["symbols"][0]
                self.memory["last_symbol"] = symbol
                self.broker.close_position(symbol)
                self.pending_action = None
                self.pending_data = {}
                return True
            elif inp.lower().strip() == "cancel":
                self.bot_print("Action cancelled.")
                self.pending_action = None
                self.pending_data = {}
                return True
            else:
                self.bot_print("I didn't catch the symbol. Please reply with something like 'EURUSD' or type 'cancel'.")
                return False
        
        return False

    def _get_intent_response(self, tag: str, live_data=None):
        intent_data = next((item for item in self.intents_data.get("intents", []) if item['tag'] == tag), None)
        response_template = random.choice(intent_data['responses']) if intent_data and intent_data.get('responses') else None

        if isinstance(live_data, dict):
            if response_template:
                try:
                    self.bot_print(response_template.format(**live_data))
                    return
                except KeyError:
                    pass
            for key, value in live_data.items():
                self.bot_print(f"{key}: {value}")
            return

        if isinstance(live_data, str):
            self.bot_print(live_data)
            return

        if response_template:
            self.bot_print(response_template)
        elif live_data is not None:
            self.bot_print(str(live_data))

    def _execute_action(self, tag: str, inp: str, entities: dict = None):
        """Executes the action for a given intent, with multi-turn dialogue support."""
        if entities is None:
            entities = self.extract_entities(inp)
        
        # === Check if we need to ask for missing entities ===
        if tag == "execute_trade":
            symbol = entities["symbols"][0] if entities["symbols"] else self.memory.get("last_symbol")
            
            if not symbol:
                self.bot_print("Which symbol would you like to trade? (e.g., EURUSD)")
                self.pending_action = "awaiting_trade_symbol"
                self.pending_data = {"intent": tag}
                return
            
            self.bot_print(f"Thinking... Analyzing {symbol}...")
            
            # 1. Ask the Strategy Engine for its opinion
            signal = self.strategy_manager.check_signals(symbol)
            
            # 2. Ask the Risk Manager if it's safe
            portfolio_size = len(self.trading_symbols) if self.trading_symbols else 1
            allowed, risk_reason = self.risk_manager.is_trading_allowed(symbol, self.max_daily_loss, portfolio_size)
            
            user_direction = entities.get("direction")
            strategy_direction = signal.get('action', 'BUY')
            
            # 1. Determine direction
            order_type = user_direction if user_direction else strategy_direction
            
            
            # 3. Formulate Reasoning
            if not allowed:
                self.bot_print(f"🛑 I refuse to execute this trade. Reason: {risk_reason}")
                return
            
            if user_direction and strategy_direction != user_direction:
                self.bot_print(f"⚠️ Warning: My technical analysis suggests {strategy_direction}, but you want to {user_direction}.")
                override = input(f"[Bot]: Do you want to override my advice and execute a {user_direction} anyway? (y/n): ")
                if override.lower() not in ['y', 'yes']:
                    self.bot_print("Trade aborted. Good call.")
                    return
            
            if signal['action'] != 'BUY': # Assuming user asked to buy
                self.bot_print(f"⚠️ Warning: My technical analysis suggests {signal['action']}, but you want to BUY.")
                override = input("[Bot]: Do you want to override my advice and execute anyway? (y/n): ")
                if override.lower() != 'y':
                    self.bot_print("Trade aborted. Good call.")
                    return
                    
            self.bot_print(f"Logic checks passed. Executing trade for {symbol}.")
            
            # 2. Calculate safe lot size using Risk Manager
            trade_plan = self.risk_manager.calculate_safe_trade(
                symbol=symbol,
                base_risk_pct=self.risk_percentage,
                stop_loss_pips=self.stop_loss,
                max_daily_loss=self.max_daily_loss,
                portfolio_size=len(self.portfolio_manager.master_watchlist)
            )
            proposed_lots = trade_plan.get('lots', 0.01)  # Fallback to micro-lot if calculation fails
            
            # 3. Attempt the trade
            result = self.broker.execute_trade(
                symbol=symbol,
                order_type=order_type,
                lots=proposed_lots,
                stop_loss_pips=self.stop_loss,
                take_profit_pips=self.target_profit
            )
            
            if result and result.get("success"):
                self.bot_print(f"✅ Trade placed successfully! Ticket #{result.get('ticket')}")
            else:
                error_msg = result.get("comment", "Unknown Broker Error") if result else "Execution Failed"
                self.bot_print(f"❌ MT5 rejected the order. Reason: {error_msg}")
                
                # === THE REASONING FACULTY (Self-Correction) ===
                error_lower = error_msg.lower()
                
                if "margin" in error_lower or "money" in error_lower:
                    self.bot_print(f"💡 Reasoning: You don't have enough free margin to trade {proposed_lots} lots.")
                    self.bot_print("Would you like me to recalculate using the absolute minimum micro-lot (0.01) and try again? (y/n)")
                    
                    self.pending_action = "retry_micro_lot"
                    self.pending_data = {
                        "symbol": symbol,
                        "order_type": order_type
                    }
                    
                elif "market closed" in error_lower or "off quotes" in error_lower:
                    self.bot_print("💡 Reasoning: The market for {symbol} is currently closed or illiquid.")
                    self.bot_print("I cannot execute this now. Please try again during active market hours.")
                else:
                    self.bot_print("💡 Reasoning: I cannot autonomously resolve this error. Please check MT5.")
             
        
        elif tag == "close_position":
            # Find the symbol from the user's text, or remember the last one we talked about
            symbol = entities["symbols"][0] if entities["symbols"] else self.memory.get("last_symbol")
            
            if not symbol:
                self.bot_print("Which symbol would you like to close? (e.g., EURUSD)")
                self.pending_action = "awaiting_close_symbol"
                self.pending_data = {"intent": tag}
                return
            
            self.broker.close_position(symbol)
            return
        
        elif tag == "deposit" or tag == "withdraw":
            # Deposit/Withdraw need an amount
            amount = entities["money"][0] if entities["money"] else self.memory["last_money_amount"]
            if not amount:
                self.bot_print("How much would you like to transfer? (e.g., 100, 500.50)")
                self.pending_action = "awaiting_amount"
                self.pending_data = {"intent": tag}
                return
        
        # === Execute the action ===
        live_data = None
        if tag in self.action_mappings:
            live_data = self.action_mappings[tag](inp)

        if live_data is not None:
            self._get_intent_response(tag, live_data)
        elif tag not in self.action_mappings:
            self._get_intent_response(tag)

        self.last_intent = tag
        self._log_interaction(inp, tag)

    # --- System Methods ---

    def start_chat(self):
        """Establish broker connection and start the main chat loop."""
        # === BROKER CONNECTION PHASE ===
        print("")
        
        data = self._read_json(self.PROFILE_FILE)
        login = data.get("login")
        password = data.get("password")
        server = data.get("server")
        
        if (login and password and server):
            if not self.broker.connect(login=login, password=password, server=server):
                self.bot_print("⚠️ Saved credentials failed. Please log in again.")
                login = password = server = None
        
        while not self.broker.connected:
            try:
                # Validate login input
                login_input = int(input("[Bot]: Enter your MT5 Account Number: ").strip())
            except ValueError:
                self.bot_print("Account Number must be a valid integer. Try again.")
                continue
                
            if len(str(login_input)) < 5:
                self.bot_print("Account number too short. Try again.")
                continue
            
            # Get password (hidden input)
            password_input = getpass.getpass("[Bot]: Enter your MT5 Password: ")
            if not password_input:
                self.bot_print("Password cannot be empty. Try again.")
                continue

            # Get server
            server_input = input("[Bot]: Enter Broker Server (e.g., MetaQuotes-Demo): ").strip()
            if not server_input:
                self.bot_print("Server cannot be empty. Try again.")
                continue
            
            # Attempt connection
            if self.broker.connect(login=login_input, password=password_input, server=server_input):
                # Save credentials
                data.update({
                    "login": login_input,
                    "password": password_input,
                    "server": server_input
                })
                self._write_json(self.PROFILE_FILE, data)
                self.bot_print(f"✅ Successfully connected! Account #{self.broker.login} is ready.")
            else:
                self.bot_print("❌ Connection failed. Please verify your credentials and ensure MT5 is running.")
        
        # === TRADING CONFIG PHASE ===
        if not self._load_trading_config():
            self.setup_trading_config()
        else:
            # Clean, non-blocking startup dashboard
            print("\n" + "="*50)
            print(f"🤖 SYSTEM ONLINE | Risk: {self.risk_percentage}% | TP: ${self.target_profit} | SL: {self.stop_loss} pips")
            if self.trading_symbols:
                print(f"📊 Tracking ({len(self.trading_symbols)}): {', '.join(self.trading_symbols)}")
            print("="*50)
        
        # === MAIN CHAT LOOP ===
        print("\n[Bot]: System ready! Type 'scan' to hunt, or ask me to change your settings.")
        try:
            while True:
                # Use patch_stdout to keep the prompt safe from background prints
                with patch_stdout():
                    try:
                        inp = self.session.prompt("[You]: ").strip()
                        follow_up_match = re.search(r"(?:no,?\s*)?(?:make|change|set) it (?:to\s*)?([0-9.]+)", inp, re.IGNORECASE)
                        match = re.search(r"change (?:my )?(.*?)\s+to\s+([0-9.]+)", inp, re.IGNORECASE)

                        target_key = None
                        new_value = None
                        if follow_up_match:
                            # Check if the bot remembers what we were talking about
                            if hasattr(self, 'memory') and "last_setting" in self.memory:
                                target_key = self.memory["last_setting"]
                                new_value = float(follow_up_match.group(1))
                        if match:
                            target_key = match.group(1).strip()
                            new_value = float(match.group(2))
                        
                        if target_key and new_value is not None:
                            # Extract the key and value
                            target_key = match.group(1).strip()
                            new_value = float(match.group(2))
                            
                            # Map the spoken {key} to your bot's internal variables
                            if target_key in ["daily loss", "daily loss limit", "max loss limit"]:
                                self.max_daily_loss = new_value
                                display_name = "Max Daily Loss Limit"
                                
                            elif target_key in ["risk", "risk percentage", "risk per trade"]:
                                self.risk_percentage = new_value
                                display_name = "Risk Per Trade"
                                
                            elif target_key in ["target", "target profit", "profit target"]:
                                self.target_profit = new_value
                                display_name = "Target Profit"
                            elif target_key in ['stop loss', 'sl', 'stop-loss', 'stoploss']:
                                self.stop_loss = new_value
                                display_name = "Stop Loss"
                            else:
                                self.bot_print(f"I don't recognize the setting key '{target_key}'. Valid keys are: risk, target, daily loss, stop loss.")
                                continue # Skip the rest of the loop and wait for next input

                            if not hasattr(self, 'memory'):
                                self.memory = {}
                            self.memory["last_setting"] = target_key
                            # Save to profile.json
                            self._save_trading_config()
                            self.bot_print(f"✅ I have updated your {display_name} to {new_value}.")
                            
                            # Skip the neural network since we've already handled the command
                            continue 

                    except EOFError:
                        # Handle Ctrl+D (EOF)
                        print("\n[Bot]: Initiating graceful shutdown...")
                        break
                
                if not inp:
                    continue
                if inp.lower() == "quit":
                    print("\n[Bot]: Initiating graceful shutdown...")
                    break
                self._handle_intent(inp)
        except KeyboardInterrupt:
            print("\n[Bot]: Interrupt detected during chat.")
        except Exception as e:
            print(f"\n[Bot]: ❌ Error during chat: {e}")
