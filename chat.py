import json
import random
import re
import sys
import time
import getpass
import threading
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional
from prompt_toolkit.patch_stdout import patch_stdout

from manager.nlp_engine import NLPEngine
from manager.reasoning_engine import ReasoningEngine
from manager.response_engine import ResponseEngine
from manager.portfolio_manager import PortfolioManager
from manager.risk_manager import RiskManager
from manager.profile_manager import ProfileManager
from strategies.strategy_manager import StrategyManager
from trader import Trader
from rich.console import Console
from prompt_toolkit import PromptSession

class ARIA:
    """Deterministic chatbot using NLP -> Reasoning -> Response pipeline."""

    def __init__(
        self,
        intents_filepath: str,
        broker: Trader,
        strategy_manager: StrategyManager,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
    ):
        # Initialize components
        self.nlp = NLPEngine(intents_filepath=intents_filepath, data_dir=str(Path("data")))
        self.profile_manager = ProfileManager(data_dir=str(Path("data")))

        self.broker = broker
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager

        self.reasoning = ReasoningEngine(
            self.strategy_manager,
            self.risk_manager,
            self.portfolio_manager
        )
        self.responder = ResponseEngine()

        # Memory for context
        self.memory = {
            "last_symbol": None,
            "last_signal": None,
        }

        self.console = Console()
        self.session = PromptSession()

        # Add notification inbox for compatibility
        self.notification_inbox: list = []
        self.inbox_lock = threading.Lock()

    def receive_system_alert(self, msg: str, priority: str = "normal"):
        """Receive system alerts (for compatibility with main.py)."""
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        if priority in ["critical", "trade_executed"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] [ARIA]: 🚨 {msg}" if priority == "critical" else f"\n[{timestamp}] [ARIA]: ✅ {msg}")

    def start_chat(self):
        """Simplified chat startup without Gemini."""
        print("")

        # Load or request broker credentials
        data = self.profile_manager._read_json(self.profile_manager.PROFILE_FILE)
        login = data.get("login")
        password = data.get("password")
        server = data.get("server")

        if login and password and server:
            if not self.broker.connect(login=login, password=password, server=server):
                self._type_print("Saved credentials failed — please log in again.")
                login = password = server = None

        while not self.broker.connected:
            try:
                login_input = int(input("[ARIA]: MT5 Account Number: ").strip())
            except ValueError:
                self._type_print("Account number must be an integer.")
                continue
            if len(str(login_input)) < 5:
                self._type_print("Account number too short.")
                continue

            password_input = getpass.getpass("[ARIA]: Password: ")
            if not password_input:
                self._type_print("Password can't be empty.")
                continue

            server_input = input("[ARIA]: Broker Server (e.g. MetaQuotes-Demo): ").strip()
            if not server_input:
                self._type_print("Server can't be empty.")
                continue

            if self.broker.connect(login=login_input, password=password_input, server=server_input):
                data.update({"login": login_input, "password": password_input, "server": server_input})
                self.profile_manager._write_json(self.profile_manager.PROFILE_FILE, data)
                self._type_print(f"Connected. Account #{self.broker.login} is live.")
            else:
                self._type_print("Connection failed — verify credentials and make sure MT5 is running.")

        # Simple greeting
        greeting = self.process_message("hello")
        self._type_print(greeting)

        # Main chat loop
        try:
            while True:
                with patch_stdout():
                    inp = self.session.prompt("[You]: ").strip()
                    if inp.lower() in ['quit', 'exit', 'bye']:
                        self._type_print("Goodbye!")
                        break
                    response = self.process_message(inp)
                    self._type_print(response)
        except KeyboardInterrupt:
            self._type_print("Shutting down...")

    def process_message(self, user_input: str) -> str:
        """Main processing pipeline: NLP -> Reasoning -> Response."""
        # 1. NLP classification
        intent_data = self.nlp.process(user_input)
        intent = intent_data.get('intent', 'unknown')
        entities = intent_data.get('entities', [])

        # 2. Rule-Based Reasoning based on Intent
        if intent == "analyze_symbol":
            symbol = entities[0] if entities else self.memory.get("last_symbol", "EURUSD")
            if symbol:
                self.memory["last_symbol"] = symbol
                analysis = self.reasoning.analyze_asset(symbol)

                # Generate comprehensive analysis response
                narration = analysis.get("narration", "")
                regime_advice = analysis.get("regime_advice", "")
                scored = analysis.get("scored", {})
                grade = scored.get("grade", "C")

                response = f"{narration} {regime_advice}"
                if grade in ["A", "B"]:
                    response += f" Quality grade: {grade} — this setup has strong potential."
                elif grade == "C":
                    response += f" Quality grade: {grade} — proceed with caution."
                else:
                    response += f" Quality grade: {grade} — I'd recommend waiting for a better setup."

                return response

        elif intent == "portfolio_status":
            context = self.reasoning.get_portfolio_context()
            return self.responder.action_response(
                f"Portfolio valued at ${context.get('total_value', '0.00')} "
                f"with {context.get('active_positions', 0)} active positions. "
                f"Daily P&L: {context.get('daily_pnl', '+0.00')}.",
                "data_returned"
            )

        elif intent == "greeting":
            # Get account info for greeting
            account = self.broker.getAccountInfo() or {}
            suggestion = "Ready for analysis."
            return self.responder.greeting(account, suggestion)

        # 3. Fallback for unknown intents
        return self.responder.fallback()

    def chat(self, user_message: str, action_result: str = None) -> str:
        """Route to the new deterministic processing pipeline."""
        if action_result:
            return f"Action completed: {action_result}. How can I assist you further?"
        else:
            return self.process_message(user_message)

    # ── Output helpers ────────────────────────────────────────────────────────

    def _type_print(self, msg: str, delay: float = 0.012) -> None:
        """
        Typewriter effect — characters print one by one for a human-feeling output.
        Falls back to plain print if stdout isn't a real TTY (CI, redirected).
        """
        timestamp  = datetime.now().strftime("%H:%M:%S")
        clean_msg  = msg.replace("[Bot]: ", "").replace("[Bot]:", "").strip()
        prefix     = f"[{timestamp}] [ARIA]: "

        if sys.stdout.isatty():
            sys.stdout.write(prefix)
            for char in clean_msg:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay)
            sys.stdout.write("\n")
        else:
            print(f"{prefix}{clean_msg}")

    def bot_print(self, msg: str) -> None:
        """Alias kept for compatibility — routes through _type_print."""
        self._type_print(msg)

    def _think_step(self, step: int, text: str) -> None:
        """Displays a numbered reasoning step with dim styling."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [ARIA ▸ {step}]: {text}")

    def _proactive_suggest(self, suggestion: str) -> None:
        """Prints a soft follow-up nudge after completing an action."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [ARIA 💡]: {suggestion}")

    # ── Live context snapshot ─────────────────────────────────────────────────

    def _build_context_snapshot(self) -> dict:
        """
        Assembles a concise summary of the current trading state.
        """
        ctx: dict = {}

        # Account data
        if self.broker.connected:
            acct = self.broker.getAccountInfo()
            if acct:
                ctx["Account Balance"]  = f"${acct.balance:,.2f}"
                ctx["Account Equity"]   = f"${acct.equity:,.2f}"
                ctx["Floating P&L"]     = f"${acct.profit:,.2f}"
                ctx["Margin Level"]     = f"{acct.margin_level:.1f}%" if acct.margin_level else "N/A"

            positions = self.broker.getPositions()
            if positions:
                pos_summary = ", ".join(
                    f"{p.symbol} {'BUY' if p.type == 0 else 'SELL'} "
                    f"{p.volume}L @ {p.price_open} (P&L ${p.profit:,.2f})"
                    for p in positions
                )
                ctx["Open Positions"] = pos_summary
            else:
                ctx["Open Positions"] = "None"
        else:
            ctx["Broker"] = "Disconnected"

        # Config
        ctx["Tracked Symbols"]  = ", ".join(self.trading_symbols) if self.trading_symbols else "None"
        ctx["Risk Per Trade"]   = f"{self.risk_percentage}%"
        ctx["Max Daily Loss"]   = f"${self.max_daily_loss}"
        ctx["Target Profit"]    = f"${self.target_profit}"

        # Short-term memory
        if self.memory.get("last_symbol"):
            ctx["Last Discussed Symbol"] = self.memory["last_symbol"]
        if self.memory.get("last_signal"):
            sig = self.memory["last_signal"]
            ctx["Last Signal"] = (
                f"{sig.get('action')} @ {sig.get('confidence', 0):.0%} confidence — "
                f"{sig.get('reason', '')}"
            )

        return ctx

    # ── IO helpers ────────────────────────────────────────────────────────────

    def _log_interaction(self, user_input: str, intent: str, status: str = "completed"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "intent": intent,
            "status": status,
        }
        history = self._read_json(self.HISTORY_FILE, [])
        history.append(entry)
        self._write_json(self.HISTORY_FILE, history[-200:])

    def _read_json(self, filepath: Path, default: Any = None) -> Any:
        if not filepath.exists():
            return default if default is not None else {}
        try:
            with filepath.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default if default is not None else {}

    def _write_json(self, filepath: Path, data: Any):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as f:
            json.dump(data, f, indent=4)

    def receive_system_alert(self, msg: str, priority: str = "normal"):
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        if priority in ["critical", "trade_executed"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            with patch_stdout():
                icon = "🚨" if priority == "critical" else "✅"
                print(f"\n[{timestamp}] [ARIA]: {icon} {msg}")

    def _read_inbox(self):
        with self.inbox_lock:
            if not self.notification_inbox:
                return "Inbox is clean — nothing new from the scanner."
            items = list(self.notification_inbox)
            self.notification_inbox.clear()

        lines = [f"You have {len(items)} unread event(s):"]
        for item in items:
            icon = {"trade_executed": "🟢", "critical": "⚠️"}.get(item["priority"], "ℹ️")
            lines.append(f"  {icon} {item['msg']}")
        return "\n".join(lines)

    def _get_validated_input(self, prompt: str, cast_type: type,
                             validation_func: Callable, error_msg: str) -> Any:
        while True:
            raw = input(prompt).strip()
            if not raw:
                continue
            try:
                val = (
                    [i.strip().upper() for i in raw.split(",")]
                    if cast_type is list else cast_type(raw)
                )
                if validation_func(val):
                    return val
                self.bot_print(error_msg)
            except ValueError:
                self.bot_print(f"Please enter a valid {cast_type.__name__}.")

    # ── Symbol helpers ────────────────────────────────────────────────────────

    def _symbol_file_path(self, symbol: str) -> Path:
        return self.SYMBOLS_CACHE_DIR / f"{symbol.upper()}.json"

    def _load_local_symbols(self) -> list:
        symbols = []
        if not self.SYMBOLS_CACHE_DIR.exists():
            return []
        for file in self.SYMBOLS_CACHE_DIR.glob("*.json"):
            try:
                data = self._read_json(file, {})
                if isinstance(data, dict) and data.get("symbol"):
                    symbols.append(data)
            except Exception:
                continue
        self.symbol_cache  = symbols
        self.symbol_lookup = {item["symbol"].upper(): item for item in symbols}
        return symbols

    def _save_local_symbol(self, symbol_data: dict):
        self.SYMBOLS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._write_json(self._symbol_file_path(symbol_data.get("symbol", "UNKNOWN")), symbol_data)
        self.symbol_cache = [
            s for s in self.symbol_cache
            if s.get("symbol", "").upper() != symbol_data.get("symbol", "").upper()
        ]
        self.symbol_cache.append(symbol_data)
        self.symbol_lookup[symbol_data.get("symbol", "").upper()] = symbol_data

    def _refresh_symbol_cache(self) -> list:
        symbols = self.broker.getSymbols()
        if not symbols:
            return self._load_local_symbols()
        symbol_data = []
        for s in symbols:
            name = getattr(s, "name", None) or getattr(s, "symbol", None)
            if name:
                symbol_data.append({
                    "symbol":      name,
                    "description": getattr(s, "description", ""),
                    "bid":         getattr(s, "bid", 0),
                    "ask":         getattr(s, "ask", 0),
                })
        self.symbol_cache  = symbol_data
        self.symbol_lookup = {item["symbol"].upper(): item for item in symbol_data}
        return symbol_data

    def _get_symbol_cache(self) -> list:
        if self.symbol_cache:
            return self.symbol_cache
        cached = self._load_local_symbols()
        return cached if cached else self._refresh_symbol_cache()

    def _infer_asset_class(self, symbol: str) -> str:
        symbol = symbol.upper()
        if any(symbol.startswith(p) for p in ["XAU", "XAG", "XPT", "XPD"]):
            return "Metals"
        if any(t in symbol for t in ["BTC", "ETH", "LTC", "XBT", "USDT", "DOGE"]):
            return "Crypto"
        return "Forex"

    def _add_symbols_to_portfolio(self, symbols: list) -> list:
        added = []
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol not in self.trading_symbols:
                self.trading_symbols.append(symbol)
                added.append(symbol)
            if hasattr(self.portfolio_manager, "add_symbol"):
                self.portfolio_manager.add_symbol(
                    symbol, asset_class=self._infer_asset_class(symbol)
                )
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
        self.risk_manager.max_open_trades = max(1, len(self.trading_symbols))

    def _parse_timeframe_from_text(self, text: str) -> str:
        match = re.search(r"\b(M1|M5|M15|M30|H1|H4|D1)\b", text.upper())
        return match.group(1) if match else "H1"

    def _normalize_ohlcv_dataframe(self, df):
        if df is None or df.empty:
            return df
        if "time" in df.columns:
            try:
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
                df = df.drop(columns=["time"], errors="ignore")
            except Exception:
                pass
        vol_col = next((c for c in ["volume", "tick_volume", "real_volume"] if c in df.columns), None)
        keep    = ["datetime", "open", "high", "low", "close"]
        if vol_col:
            keep.append(vol_col)
            df = df[keep].rename(columns={vol_col: "volume"})
        else:
            df = df[[c for c in keep if c in df.columns]]
        return df

    def _download_portfolio_ohlcv(self, timeframe: str = "H1", count: int = 200):
        symbols = self._get_portfolio_symbols()
        if not symbols:
            self.bot_print("No portfolio symbols found. Add some symbols first.")
            return
        if not self.broker.connected:
            self.bot_print("MT5 needs to be connected to download OHLCV data.")
            return
        output_dir = self.DATA_DIR / "ohlcv"
        output_dir.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            self.bot_print(f"Downloading {symbol} ({timeframe})...")
            df = self.broker.get_historical_rates(symbol, timeframe=timeframe, count=count)
            if df is None or df.empty:
                self.bot_print(f"No data for {symbol}.")
                continue
            df = self._normalize_ohlcv_dataframe(df)
            if df is not None and not df.empty:
                df.to_csv(output_dir / f"{symbol}.csv", index=False)
                self.bot_print(f"Saved {symbol}.csv")
        self.bot_print("Portfolio OHLCV download complete.")

    # ── Symbol command handler ────────────────────────────────────────────────

    def _handle_symbol_command(self, inp: str) -> bool:
        clean = inp.lower().strip()

        ohlcv_keywords = [
            "generate ohlc", "generate ohlcv", "download ohlc", "download ohlcv",
            "export ohlc", "export ohlcv", "download csv", "export csv",
        ]
        if any(k in clean for k in ohlcv_keywords) or (
            any(w in clean for w in ["generate", "download"]) and "portfolio" in clean
        ):
            self._download_portfolio_ohlcv(timeframe=self._parse_timeframe_from_text(inp))
            return True

        if any(k in clean for k in ["available symbols", "show symbols", "list symbols",
                                    "my symbols", "portfolio symbols"]):
            self._show_available_symbols()
            return True

        if any(k in clean for k in ["add to portfolio", "add to watchlist", "add symbol",
                                    "track", "watchlist"]) or (
            "add" in clean and "portfolio" in clean
        ):
            entities = self.extract_entities(inp)
            symbols  = entities["symbols"]
            if not symbols:
                self.bot_print("Which symbol? Try 'Add AUDCAD to portfolio'.")
                return True
            added = self._add_symbols_to_portfolio(symbols)
            result_msg = (
                f"Added {', '.join(added)} to your watchlist."
                if added else
                f"Already tracking: {', '.join(symbols)}"
            )
            self.bot_print(self.chat(inp, action_result=result_msg))
            return True

        return False

    def _show_available_symbols(self):
        symbols = self._get_portfolio_symbols()
        if not symbols:
            self.bot_print("No symbols configured. Add some with 'Add EURUSD to portfolio'.")
            return
        self.bot_print(f"Tracking {len(symbols)} symbol(s):")
        for i, s in enumerate(symbols, 1):
            print(f"  {i}. {s}")

    # ── Config helpers ────────────────────────────────────────────────────────

    def _parse_conversational_config(self, text: str) -> dict:
        text_upper = text.upper()
        symbols    = list(set(re.findall(r"\b[A-Z]{6}\b", text_upper)))
        timeframes = list(set(re.findall(r"\b(M1|M5|M15|M30|H1|H4|D1)\b", text_upper)))
        risk_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        profit_match = re.search(
            r"(?i)(?:make|profit|target).*?\$?(\d+(?:\.\d+)?)\s*(?:daily|per day)", text
        )
        return {
            "symbols":      symbols,
            "timeframes":   timeframes,
            "risk":         float(risk_match.group(1)) if risk_match else 0.0,
            "daily_profit": float(profit_match.group(1)) if profit_match else 0.0,
        }

    def setup_trading_config(self):
        print("\n" + "=" * 40 + "\n🤖  TRADING SETUP\n" + "=" * 40)
        user_input = input("[ARIA]: Describe your goal (e.g. 'Risk 1%, make $10 daily on EURUSD'): ").strip()
        parsed = self._parse_conversational_config(user_input)

        self.trading_symbols = parsed["symbols"]
        self.risk_percentage = parsed["risk"]
        if parsed["daily_profit"] > 0:
            self.daily_goal    = parsed["daily_profit"]
            session_goal       = self.daily_goal / 10.0
            count              = len(self.trading_symbols) or 1
            self.target_profit = round(session_goal / count, 2)
            self.bot_print(f"Daily goal ${self.daily_goal}. Session target ${session_goal}, "
                           f"${self.target_profit} per symbol across {count} pairs.")

        if not self.trading_symbols:
            self.trading_symbols = self._get_validated_input(
                "[ARIA]: Symbols (e.g. EURUSD, XAUUSD): ",
                list, lambda x: len(x) > 0, "Enter at least one symbol."
            )
        if self.risk_percentage <= 0:
            self.risk_percentage = self._get_validated_input(
                "[ARIA]: Risk % per trade: ",
                float, lambda x: 0 < x <= 10, "Enter a value between 0.1 and 10."
            )
        if self.max_daily_loss <= 0:
            self.max_daily_loss = self._get_validated_input(
                "[ARIA]: Max daily loss ($): ",
                float, lambda x: x > 0, "Enter a positive number."
            )

        self._save_trading_config()
        self._update_max_open_trades()
        self.bot_print("Configuration saved. Ready to trade.")

    def _save_trading_config(self):
        config = self._read_json(self.PROFILE_FILE)
        config.update({
            "trading_symbols":      self.trading_symbols,
            "daily_goal":           self.daily_goal,
            "target_profit":        self.target_profit,
            "stop_loss":            self.stop_loss,
            "risk_percentage":      self.risk_percentage,
            "max_daily_loss":       self.max_daily_loss,
            "preferred_timeframes": self.preferred_timeframes,
        })
        self._write_json(self.PROFILE_FILE, config)

    def _load_trading_config(self) -> bool:
        config = self._read_json(self.PROFILE_FILE)
        if "trading_symbols" in config:
            self.trading_symbols      = config.get("trading_symbols", [])
            self.daily_goal           = config.get("daily_goal", 0.0)
            self.target_profit        = config.get("target_profit", 0.0)
            self.stop_loss            = config.get("stop_loss", 0.0)
            self.risk_percentage      = config.get("risk_percentage", 0.0)
            self.max_daily_loss       = config.get("max_daily_loss", 0.0)
            self.preferred_timeframes = config.get("preferred_timeframes", [])
            self._update_max_open_trades()
            return True
        return False

    def _format_symbols_data(self):
        if self.trading_symbols:
            return {"symbols": ", ".join(self.trading_symbols)}
        cache = self._get_symbol_cache()
        return {"symbols": ", ".join(i["symbol"] for i in cache[:20])} if cache else {"symbols": "None"}

    def _format_settings_data(self):
        return {
            "symbols":         ", ".join(self.trading_symbols) if self.trading_symbols else "None",
            "target_profit":   f"${self.target_profit}",
            "stop_loss":       f"{self.stop_loss} pips",
            "risk_percentage": f"{self.risk_percentage}%",
            "max_daily_loss":  f"${self.max_daily_loss}",
            "timeframes":      ", ".join(self.preferred_timeframes) or "H1",
        }

    # ── Trading actions ───────────────────────────────────────────────────────

    def _run_autonomous_scan(self):
        self.bot_print("Running portfolio scan... 🔎")
        results = self.portfolio_manager.evaluate_portfolio_opportunities(
            risk_pct=self.risk_percentage,
            stop_loss=20.0,
            max_daily_loss=self.max_daily_loss,
        )
        for r in results:
            self.bot_print(r)
        return self.portfolio_manager.get_portfolio_health()

    def _retrain_model(self):
        symbol = self.memory.get("last_symbol")
        if not symbol:
            return "No symbol in memory. Mention a symbol first, then ask me to retrain."
        self.strategy_manager.continuous_learning_routine(symbol)
        return f"Retrained model for {symbol}."

    # ── Intent handling ───────────────────────────────────────────────────────

    def _handle_intent(self, inp: str) -> bool:
        clean = inp.lower().strip()

        # Step 1: Multi-turn continuation takes priority
        if self.pending_action:
            return self._handle_pending_action(inp)

        # Step 2: Extract entities + update memory
        entities = self.extract_entities(inp)
        if entities["symbols"]:
            self.memory["last_symbol"] = entities["symbols"][0]
        if entities["timeframes"]:
            self.memory["last_timeframe"] = entities["timeframes"][0]
        if entities["money"]:
            self.memory["last_money_amount"] = entities["money"][0]

        if len(clean) <= 2 and clean not in {"hi", "yo"}:
            self.bot_print(self.chat(inp))
            return True

        # Step 3: Symbol-specific command shortcuts
        if self._handle_symbol_command(inp):
            return True

        # Step 4: Hardcoded shortcuts for high-frequency commands
        if "retrain" in clean and "model" in clean:
            self._execute_action("retrain_model", inp, entities)
            return True
        if any(w in clean for w in ["start", "scan"]):
            self._execute_action("bulk_scan", inp, entities)
            return True

        # Step 5: Build context snapshot
        self._build_context_snapshot()

        # Step 6: NLP classification
        intent_tag, confidence = self.predict_intent(inp)

        # Step 7: High confidence → execute immediately
        if confidence >= 0.75:
            self._execute_action(intent_tag, inp, entities)
            return True

        # Step 8: Medium confidence → execute the predicted action
        if 0.30 <= confidence < 0.75:
            self._execute_action(intent_tag, inp, entities)
            return True

        # Step 9: Low confidence — provide a simple response
        self.bot_print(self.chat(inp))
        return True

    # ── Multi-turn pending action handler ────────────────────────────────────

    def _handle_pending_action(self, inp: str) -> bool:
        action = self.pending_action

        if action == "awaiting_trade_symbol":
            entities = self.extract_entities(inp)
            if entities["symbols"]:
                symbol = entities["symbols"][0]
                self.memory["last_symbol"] = symbol
                self.pending_data["symbol"] = symbol
                self._execute_action(self.pending_data["intent"], f"trade {symbol}", entities)
                self.pending_action = None
                self.pending_data   = {}
                return True
            if inp.lower().strip() == "cancel":
                self.bot_print(self.chat("I cancelled the trade.", action_result="Trade cancelled by user."))
                self.pending_action = None
                self.pending_data   = {}
                return True
            self.bot_print("Still need the symbol — try something like 'EURUSD', or type 'cancel'.")
            return False

        if action == "awaiting_amount":
            entities = self.extract_entities(inp)
            if entities["money"]:
                amount = entities["money"][0]
                self.memory["last_money_amount"] = amount
                self.pending_data["amount"] = amount
                self._execute_action(self.pending_data["intent"], inp, entities)
                self.pending_action = None
                self.pending_data   = {}
                return True
            if inp.lower().strip() == "cancel":
                self.bot_print("No problem, cancelled.")
                self.pending_action = None
                self.pending_data   = {}
                return True
            self.bot_print("Give me a number — like '100' or '50.5', or type 'cancel'.")
            return False

        if action == "confirm_learning":
            if inp.lower() in {"y", "yes"}:
                tag      = self.pending_data.get("tag", "")
                pattern  = self.pending_data.get("original_input", "")
                response = self.pending_data.get("response", "")
                self.add_intent_pattern(tag, pattern, notify_callback=self.receive_system_alert)
                self.bot_print("Knowledge updated. I'll handle that better next time.")
            else:
                self.bot_print("Got it, skipping.")
            self.pending_action = None
            self.pending_data   = {}
            return True

        if action == "teach_new_phrase":
            self.bot_print("Let's start fresh — what would you like to do?")
            self.pending_action = None
            self.pending_data   = {}
            return True

        if action == "retry_micro_lot":
            if inp.lower() in {"y", "yes", "yeah", "sure"}:
                symbol     = self.pending_data["symbol"]
                order_type = self.pending_data["order_type"]
                result     = self.broker.execute_trade(
                    symbol=symbol, order_type=order_type, lots=0.01,
                    stop_loss_pips=self.stop_loss, take_profit_pips=self.target_profit,
                )
                if result and result.get("success"):
                    action_summary = f"Micro-lot trade executed for {symbol}. Ticket #{result.get('ticket')}."
                    self.bot_print(self.chat(f"micro lot trade for {symbol}", action_result=action_summary))
                else:
                    err = result.get("comment", "Unknown error") if result else "Execution failed"
                    self.bot_print(f"Failed again — broker said: {err}. Probably a margin issue worth checking in MT5.")
            else:
                self.bot_print("Understood. Leaving it for now.")
            self.pending_action = None
            self.pending_data   = {}
            return True

        if action == "awaiting_close_symbol":
            entities = self.extract_entities(inp)
            if entities["symbols"]:
                symbol = entities["symbols"][0]
                self.memory["last_symbol"] = symbol
                self.broker.close_position(symbol)
                self.bot_print(self.chat(f"close {symbol}", action_result=f"Closed position on {symbol}."))
                self.pending_action = None
                self.pending_data   = {}
                return True
            if inp.lower().strip() == "cancel":
                self.bot_print("Action cancelled.")
                self.pending_action = None
                self.pending_data   = {}
                return True
            self.bot_print("Which symbol? Try 'EURUSD', or type 'cancel'.")
            return False

        return False

    # ── Action executor ───────────────────────────────────────────────────────

    def _execute_action(self, tag: str, inp: str, entities: dict = None):
        """
        Executes the action for a given intent. Provides a response after each action.
        """
        if entities is None:
            entities = self.extract_entities(inp)

        self._build_context_snapshot()   # Ensure context is current

        # ── Trade execution (most complex path) ──────────────────────────────
        if tag in {"execute_trade", "open_buy", "open_sell", "trade_execution"}:
            symbol = entities["symbols"][0] if entities["symbols"] else self.memory.get("last_symbol")

            if not symbol:
                self.bot_print("Which symbol would you like to trade?")
                self.pending_action = "awaiting_trade_symbol"
                self.pending_data   = {"intent": tag}
                return

            # Step 1: Think aloud
            self._think_step(1, f"Fetching market data and analysing {symbol}...")
            signal = self.strategy_manager.check_signals(symbol)
            self.memory["last_signal"] = signal

            # AI reasoning removed - proceeding with signal analysis

            # Step 2: Risk check
            self._think_step(3, "Checking risk rules...")
            portfolio_size    = len(self.trading_symbols) or 1
            allowed, risk_msg = self.risk_manager.is_trading_allowed(
                symbol, self.max_daily_loss, portfolio_size
            )

            if not allowed:
                block_response = self.chat(
                    inp,
                    action_result=f"Trade BLOCKED for {symbol}. Risk reason: {risk_msg}"
                )
                self.bot_print(block_response)
                return

            # Step 3: Resolve direction — user intent wins; otherwise follow signal
            user_direction     = entities.get("direction")
            strategy_direction = signal.get("action", "BUY")
            order_type         = user_direction if user_direction else strategy_direction

            # Warn if user overrides the signal (only once, not twice)
            if user_direction and user_direction != strategy_direction:
                self.bot_print(
                    f"⚠️ My analysis says {strategy_direction} on {symbol}, "
                    f"but you're going {user_direction}. Override? (y/n)"
                )
                if input("[You]: ").lower().strip() not in {"y", "yes"}:
                    self.bot_print(self.chat(inp, action_result="Trade aborted by user after direction conflict."))
                    return

            # Step 4: Size the trade
            self._think_step(4, "Calculating safe lot size...")
            trade_plan    = self.risk_manager.calculate_safe_trade(
                symbol=symbol,
                base_risk_pct=self.risk_percentage,
                stop_loss_pips=self.stop_loss,
                max_daily_loss=self.max_daily_loss,
                portfolio_size=len(self.portfolio_manager.master_watchlist),
            )
            proposed_lots = trade_plan.get("lots", 0.01)

            # Step 5: Execute
            self._think_step(5, f"Submitting {order_type} {proposed_lots}L for {symbol}...")
            result = self.broker.execute_trade(
                symbol=symbol,
                order_type=order_type,
                lots=proposed_lots,
                stop_loss_pips=self.stop_loss,
                take_profit_pips=self.target_profit,
            )

            if result and result.get("success"):
                action_summary = (
                    f"Trade executed: {order_type} {proposed_lots} lots of {symbol}. "
                    f"Ticket #{result.get('ticket')}. "
                    f"Signal confidence was {signal.get('confidence', 0):.0%}."
                )
                self.bot_print(self.chat(inp, action_result=action_summary))

                # Generate proactive suggestion using current context
                context = self._build_context_snapshot()
                regime = self.reasoning.learner.get_current_regime() if self.reasoning.learner else "Unknown"
                self._proactive_suggest(
                    self.responder.proactive(context, regime, self.reasoning)
                )
            else:
                err = result.get("comment", "Unknown broker error") if result else "Execution failed"
                self.bot_print(f"MT5 rejected the order: {err}")
                err_lower = err.lower()
                if "margin" in err_lower or "money" in err_lower:
                    self.bot_print(
                        f"Looks like a margin issue with {proposed_lots}L. "
                        f"Want me to retry with a micro-lot (0.01)? (y/n)"
                    )
                    self.pending_action = "retry_micro_lot"
                    self.pending_data   = {"symbol": symbol, "order_type": order_type}
                elif "market closed" in err_lower or "off quotes" in err_lower:
                    self.bot_print(self.chat(
                        inp,
                        action_result=f"Trade failed: {symbol} market is closed or has no quotes right now."
                    ))
            return

        # ── Close position ────────────────────────────────────────────────────
        if tag == "close_position":
            symbol = entities["symbols"][0] if entities["symbols"] else self.memory.get("last_symbol")
            if not symbol:
                self.bot_print("Which symbol should I close?")
                self.pending_action = "awaiting_close_symbol"
                self.pending_data   = {"intent": tag}
                return
            self.broker.close_position(symbol)
            self.bot_print(self.chat(inp, action_result=f"Closed {symbol} position."))
            return

        # ── Deposit / Withdraw ────────────────────────────────────────────────
        if tag in {"deposit", "withdraw"}:
            amount = entities["money"][0] if entities["money"] else self.memory["last_money_amount"]
            if not amount:
                self.bot_print("How much would you like to transfer?")
                self.pending_action = "awaiting_amount"
                self.pending_data   = {"intent": tag}
                return

        # ── Generic mapped actions ────────────────────────────────────────────
        live_data = None
        if tag in self.action_mappings:
            live_data = self.action_mappings[tag](inp)

        # Format the data and provide a response
        if isinstance(live_data, dict):
            # Format as plain key-value string for context
            data_str = " | ".join(f"{k}: {v}" for k, v in live_data.items())
            response = self.chat(inp, action_result=data_str)
            self.bot_print(response)
        elif isinstance(live_data, str) and live_data:
            response = self.chat(inp, action_result=live_data)
            self.bot_print(response)
        elif tag not in self.action_mappings:
            # Unknown tag — just have a conversation
            self.bot_print(self.chat(inp))

        self.last_intent = tag
        self._log_interaction(inp, tag)

    # ── System startup ────────────────────────────────────────────────────────

    def start_chat(self):
        print("")

        # Load or request broker credentials
        data     = self._read_json(self.PROFILE_FILE)
        login    = data.get("login")
        password = data.get("password")
        server   = data.get("server")

        if login and password and server:
            if not self.broker.connect(login=login, password=password, server=server):
                self.bot_print("Saved credentials failed — please log in again.")
                login = password = server = None

        while not self.broker.connected:
            try:
                login_input = int(input("[ARIA]: MT5 Account Number: ").strip())
            except ValueError:
                self.bot_print("Account number must be an integer.")
                continue
            if len(str(login_input)) < 5:
                self.bot_print("Account number too short.")
                continue

            password_input = getpass.getpass("[ARIA]: Password: ")
            if not password_input:
                self.bot_print("Password can't be empty.")
                continue

            server_input = input("[ARIA]: Broker Server (e.g. MetaQuotes-Demo): ").strip()
            if not server_input:
                self.bot_print("Server can't be empty.")
                continue

            if self.broker.connect(login=login_input, password=password_input, server=server_input):
                data.update({"login": login_input, "password": password_input, "server": server_input})
                self._write_json(self.PROFILE_FILE, data)
                self.bot_print(f"Connected. Account #{self.broker.login} is live.")
            else:
                self.bot_print("Connection failed — verify credentials and make sure MT5 is running.")

        # Load or set up trading config
        if not self._load_trading_config():
            self.setup_trading_config()
        else:
            print("\n" + "=" * 55)
            print(f"🤖  ARIA ONLINE | Risk: {self.risk_percentage}% | "
                  f"TP: ${self.target_profit} | SL: {self.stop_loss} pips")
            if self.trading_symbols:
                print(f"📊  Tracking ({len(self.trading_symbols)}): {', '.join(self.trading_symbols)}")
            print("=" * 55)

        # Warm up Gemini context
        self._build_context_snapshot()

        # Greet with a Gemini-generated personalised message
        greeting = self.chat("Just started up. Give me a brief, friendly 1-sentence greeting and mention what I should do first.")
        self.bot_print(greeting)

        # ── Main chat loop ─────────────────────────────────────────────────────
        try:
            while True:
                with patch_stdout():
                    try:
                        inp = self.session.prompt("[You]: ").strip()

                        # Inline setting change shortcut (e.g. "change risk to 2")
                        match = re.search(
                            r"change (?:my )?(.*?)\s+to\s+([0-9.]+)",
                            inp, re.IGNORECASE
                        )
                        follow_up = re.search(
                            r"(?:no,?\s*)?(?:make|change|set) it (?:to\s*)?([0-9.]+)",
                            inp, re.IGNORECASE
                        )

                        target_key = new_value = None
                        if follow_up and self.memory.get("last_setting"):
                            target_key = self.memory["last_setting"]
                            new_value  = float(follow_up.group(1))
                        if match:
                            target_key = match.group(1).strip()
                            new_value  = float(match.group(2))

                        if target_key and new_value is not None:
                            setting_map = {
                                "daily loss":      ("max_daily_loss",   "Max Daily Loss"),
                                "daily loss limit":("max_daily_loss",   "Max Daily Loss"),
                                "max loss limit":  ("max_daily_loss",   "Max Daily Loss"),
                                "risk":            ("risk_percentage",  "Risk Per Trade"),
                                "risk percentage": ("risk_percentage",  "Risk Per Trade"),
                                "risk per trade":  ("risk_percentage",  "Risk Per Trade"),
                                "target":          ("target_profit",    "Target Profit"),
                                "target profit":   ("target_profit",    "Target Profit"),
                                "profit target":   ("target_profit",    "Target Profit"),
                                "stop loss":       ("stop_loss",        "Stop Loss"),
                                "sl":              ("stop_loss",        "Stop Loss"),
                            }
                            key_lower = target_key.lower()
                            if key_lower in setting_map:
                                attr, display = setting_map[key_lower]
                                setattr(self, attr, new_value)
                                self.memory["last_setting"] = key_lower
                                self._save_trading_config()
                                action_result = f"Updated {display} to {new_value}."
                                self.bot_print(self.chat(inp, action_result=action_result))
                            else:
                                self.bot_print(
                                    f"I don't recognise '{target_key}'. "
                                    f"Valid settings: risk, target, stop loss, daily loss."
                                )
                            continue

                    except EOFError:
                        print("\n[ARIA]: Shutting down gracefully...")
                        break

                if not inp:
                    continue
                if inp.lower() == "quit":
                    farewell = self.chat("User is logging out. Give a brief, warm sign-off message.")
                    self.bot_print(farewell)
                    break

                self._handle_intent(inp)

        except KeyboardInterrupt:
            print("\n[ARIA]: Interrupt received — shutting down.")
        except Exception as exc:
            print(f"\n[ARIA]: ❌ Unexpected error: {exc}")