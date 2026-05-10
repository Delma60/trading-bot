import json
import re
import sys
import time
import getpass
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from manager.market_sessions import MarketSessionManager
from manager.nlp_engine import NLPEngine
from manager.reasoning_engine import ReasoningEngine
from manager.response_engine import ResponseEngine
from manager.portfolio_manager import PortfolioManager
from manager.risk_manager import RiskManager, TradeGatekeeper, DynamicRiskTargeter, TrailingStopManager, ProfitGuard
from manager.profile_manager import ProfileManager
from manager.agent_core import (AgentPlan, AgentCore)
from manager.working_memory import WorkingMemory, ConversationTurn
from manager.episodic_memory import EpisodicMemory, Episode
from manager.user_model import UserModel
from manager.inner_monologue import InnerMonologue
from manager.voice_layer import VoiceLayer
from manager.proactive_engine import ProactiveEngine
from strategies.strategy_manager import StrategyManager
from trader import Trader
from manager.profile_manager import profile



# ─────────────────────────────────────────────────────────────────────────────
# ActionExecutor — handles broker operations recommended by the agent
# ─────────────────────────────────────────────────────────────────────────────

class ActionExecutor:
    """
    Performs real broker actions after the agent recommends them.
    Separating this from AgentCore keeps the reasoning layer side-effect-free.
    """

    def __init__(self, broker: Trader, strategy_manager: StrategyManager,
                 risk_manager: RiskManager, profile_path: Path):
        self.broker    = broker
        self.sm        = strategy_manager
        self.rm        = risk_manager
        self.profile   = profile_path

    def _cfg(self) -> dict:
        try:
            return json.loads(self.profile.read_text())
        except Exception:
            return {
                "risk_percentage": 1.0,
                "stop_loss": 20.0,
                "max_daily_loss": 500.0,
                "trading_symbols": ["EURUSD"],
                "target_profit": 10.0,
            }

    # Inside ActionExecutor in chat.py
    
    def _execute_trade_action(self, plan: AgentPlan):
        """Executes a live trade using the dynamically calculated risk targets."""
        symbol = plan.symbol
        
        # 1. Determine direction from the suggested action or context
        direction = plan.context.get("action", "BUY").upper()
        
        # 2. Calculate position size (assuming you have a method for this)
        lot_size = self.rm.calculate_position_size(symbol) 
        
        # 3. Extract the Dynamic Targets we built in the reasoning engine
        dynamic_targets = plan.context.get("dynamic_targets", {})
        b = profile.broker()
        s = profile.sessions()
        gk = TradeGatekeeper(
            max_spread_pips       = b.spread_tolerance_pips,
            avoid_asian_session   = s.avoid_asian_session,
            avoid_friday_close    = s.avoid_friday_close,
        )
        ok, reason = gk.gate(symbol, None)   # broker not needed; uses mt5 directly
        if not ok:
            return f"Trade blocked by gate: {reason}"
        
        if direction == "BUY":
            sl_pips = dynamic_targets.get("sl_buy_pips", 50)  # Safe fallback if empty
            tp_pips = dynamic_targets.get("tp_buy_pips", 100)
        elif direction == "SELL":
            sl_pips = dynamic_targets.get("sl_sell_pips", 50)
            tp_pips = dynamic_targets.get("tp_sell_pips", 100)
        else:
            return "Trade execution aborted: Invalid direction."

        # Optional: Print to console so you can physically see the dynamic targets being used
        print(f"[Execution Layer] Routing {direction} on {symbol} | SL: {sl_pips}p | TP: {tp_pips}p")

        # 4. Send the order to the MT5 Broker
        try:
            result = self.broker.execute_trade(
                symbol=symbol,
                action=direction,
                lots=lot_size,
                stop_loss_pips=sl_pips,
                take_profit_pips=tp_pips,
            )
            if result.get("success"):
                return f"Successfully executed {direction} on {symbol} (Ticket #{result['ticket']})."
            return f"Broker rejected the {direction} order for {symbol}: {result.get('reason', 'unknown error')}"
        except Exception as e:
            return f"Execution failed: {str(e)}"


    def execute_trade(self, symbol: str, direction: str, lots: Optional[float] = None, strategy: str = "Manual") -> str:
        """Execute a trade and return a human-readable outcome."""
        cfg = self._cfg()

        # Validate direction
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            return f"Invalid direction '{direction}' — must be BUY or SELL."

        # Position sizing if lots not provided
        if not lots:
            plan = self.rm.calculate_safe_trade(
                symbol         = symbol,
                base_risk_pct  = cfg.get("risk_percentage", 1.0),
                stop_loss_pips = cfg.get("stop_loss", 20.0),
                max_daily_loss = cfg.get("max_daily_loss", 500.0),
                portfolio_size = max(len(cfg.get("trading_symbols", [])), 1),
            )
            if not plan.get("approved"):
                return f"Trade blocked — {plan.get('reason', 'risk check failed')}."
            lots = plan.get("lots", 0.01)

        result = self.broker.execute_trade(
            symbol           = symbol,
            action           = direction,
            lots             = lots,
            stop_loss_pips   = cfg.get("stop_loss", 20.0),
            take_profit_pips = cfg.get("target_profit", 10.0),
            strategy         = strategy,
        )

        if result.get("success"):
            return (
                f"✅ {direction} {lots} lots of {symbol} executed. "
                f"Ticket #{result['ticket']} @ {result['price']}."
            )
        return f"❌ Order rejected: {result.get('reason', 'unknown error')}."

    def close_position(self, symbol: str) -> str:
        success = self.broker.close_position(symbol)
        return f"✅ {symbol} position closed." if success else f"❌ Failed to close {symbol}."

    def close_all(self) -> str:
        self.broker.close_all_positions()
        return "✅ All positions closed."

    def retrain(self, symbol: str) -> str:
        try:
            self.sm.continuous_learning_routine(symbol)
            return f"✅ Model retrained for {symbol}."
        except Exception as e:
            return f"Retraining failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# ARIA  —  main chatbot class
# ─────────────────────────────────────────────────────────────────────────────

class ARIA:
    """
    Adaptive Reasoning & Intelligence for Algo-trading.

    Wires NLPEngine, AgentCore, and ActionExecutor into a terminal chatbot.
    """

    PROFILE_FILE = Path("data/profile.json")

    # ── Shortcut keyword → intent overrides (before NLP) ─────────────────────
    _KEYWORD_INTENTS: list[tuple[list[str], str]] = [
        (["portfolio", "my symbols", "what am i trading", "trading symbols", "what pairs"], "trading_symbols"),
        (["add to portfolio", "add symbol", "add pair", "add", "add new", "trade new"], "add_symbols"),
        (["scan", "start scanning", "run scan", "go"], "bulk_scan"),
        (["close all profitable", "close profitable", "take profits", "take profit", "close winners", "close good positions"], "close_profitable_positions"),
        (["positions", "open trades", "what am I in", "good positions", "profitable positions", "winning trades", "winners"], "profitable_positions"),
        (["balance", "account", "equity"], "account_summary"),
        (["risk", "drawdown", "exposure"], "risk_management"),
        (["history", "pnl", "daily results", "how did I do"], "trade_history"),
        (["retrain", "update model", "relearn"], "retrain_model"),
        (["strategy", "how does", "what method"], "strategy_info"),
        (["price", "quote", "rate", "how much is"], "get_price"),
        (["close all", "nuke", "flatten"], "close_all"),
        (["what symbols", "browse symbols", "search symbols",
          "what pairs", "what can i trade", "available symbols",
          "available instruments", "what tickers", "list all",
          "show all symbols", "what metals", "what crypto",
          "what indices", "find symbol", "look up symbol",
          "what is the ticker", "what is the symbol for"], "browse_symbols"),
    ]

    def __init__(
        self,
        intents_filepath: str,
        broker: Trader,
        strategy_manager: StrategyManager,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
    ):
        # Core components
        self.broker    = broker
        self.sm        = strategy_manager
        self.pm        = portfolio_manager
        self.rm        = risk_manager

        # NLP
        self.nlp = NLPEngine(
            intents_filepath=intents_filepath,
            data_dir=str(Path("data"))
        )

        # Profile manager (credentials / config I/O)
        self.profile_manager = ProfileManager(path=Path("data/profile.json"))

        # Reasoning + response generation
        self.reasoning = ReasoningEngine(strategy_manager, risk_manager, portfolio_manager)
        self.responder = ResponseEngine()

        # Agentic core
        self.agent = AgentCore(strategy_manager, risk_manager, portfolio_manager,
                               broker, self.reasoning)

        # Action executor (side-effect layer)
        self.executor = ActionExecutor(broker, strategy_manager, risk_manager, self.PROFILE_FILE)

        # Trailing stop manager — automatically trails open positions once connected.
        self.dynamic_targeter = DynamicRiskTargeter(self.broker)
        self.trailing_manager = TrailingStopManager(self.broker, self.dynamic_targeter)
        self.trailing_manager.start()
        self.profit_guard = ProfitGuard(self.broker, notify_callback=self.receive_system_alert)
        self.profit_guard.start()
        
        # Memory — persists across turns in a single session
        self.memory: dict[str, Any] = {
            "last_symbol":    None,
            "last_signal":    None,
            "last_intent":    None,
            "last_timeframe": "H1",
            "last_setting":   None,
        }

        # ─── COGNITIVE ARCHITECTURE ───────────────────────────────────────
        # These components transform ARIA from reactive to truly conversational
        self.working_memory  = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.user_model      = UserModel()
        self.inner_monologue = InnerMonologue(
            self.working_memory, self.episodic_memory, self.user_model
        )
        self.voice_layer = VoiceLayer(
            self.working_memory, self.user_model, self.inner_monologue
        )
        self.proactive_engine = ProactiveEngine(
            broker, self.working_memory, self.episodic_memory,
            self.receive_system_alert
        )
        
        # Start proactive engine and observe session start
        self.proactive_engine.start()
        self.user_model.observe("session_start")
        # ─────────────────────────────────────────────────────────────────

        # Pending multi-turn state
        self.pending_action: Optional[str] = None
        self.pending_data: dict = {}

        # Notification inbox (filled by background scanner)
        self.notification_inbox: list = []
        self.inbox_lock = threading.Lock()
        
        # Message processing timeout (prevents hung broker calls from freezing chat)
        self.message_processor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ARIA-msg")
        self.message_timeout_seconds = 30

        self.console = Console()
        self.session = PromptSession()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — called by main.py
    # ─────────────────────────────────────────────────────────────────────────

    def receive_system_alert(self, msg: str, priority: str = "normal"):
        """Receive alerts from the background scanner thread."""
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        if priority in ("critical", "trade_executed"):
            ts   = datetime.now().strftime("%H:%M:%S")
            icon = "🚨" if priority == "critical" else "✅"
            print(f"\n[{ts}] [ARIA]: {icon} {msg}")

    def start_chat(self):
        """Connect to MT5, load config, then enter the main chat loop."""
        self._broker_login()
        self._load_or_setup_config()
        self._print_banner()

        greeting = self._make_session_aware_greeting()
        self._type_print(greeting)

        self._run_loop()

    # ─────────────────────────────────────────────────────────────────────────
    # Core message pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process_message(self, user_input: str) -> str:
        """
        Main pipeline: keyword shortcuts → NLP → COGNITIVE LAYER → agent → optional action.
        Wrapped with timeout to prevent hung broker calls from freezing the chat loop.
        
        The cognitive layer now captures:
        - User emotion
        - Conversation narrative (working memory)
        - Inner reasoning (monologue)
        - Natural voice (voice layer)
        - Behavioral patterns (user model updates)
        """
        try:
            future = self.message_processor.submit(self._process_message_impl, user_input)
            return future.result(timeout=self.message_timeout_seconds)
        except FuturesTimeoutError:
            mgr = MarketSessionManager()
            portfolio_symbols = profile.symbols()
            open_syms, closed_syms = mgr.filter_tradeable_symbols(portfolio_symbols)
            
            if closed_syms and not open_syms:
                msg = (
                    f"⏱️ Data request timed out after {self.message_timeout_seconds}s. "
                    f"Your entire portfolio ({', '.join(closed_syms[:3])}…) is currently in closed markets, "
                    f"which typically causes the broker connection to hang. Try again when markets open or switch to crypto!"
                )
            elif closed_syms:
                msg = (
                    f"⏱️ Message processing timed out after {self.message_timeout_seconds}s. "
                    f"Note: Some of your symbols ({', '.join(closed_syms[:2])}…) are in closed markets, "
                    f"which might be causing the broker to freeze on data retrieval."
                )
            else:
                msg = f"🚨 ⏱️ Message processing timed out after {self.message_timeout_seconds}s — broker may be unresponsive. Please try again."
                
            self.notify(msg, priority="critical")
            return msg
    
    def _process_message_impl(self, user_input: str) -> str:
        """Internal implementation of message processing (may be called with timeout)."""
        text = user_input.strip()
        if not text:
            return ""

        # ── 0. Pending multi-turn state takes priority ─────────────────────
        if self.pending_action:
            handled, response = self._handle_pending(text)
            if handled:
                return response

        # ── 1. Check inbox ─────────────────────────────────────────────────
        if any(k in text.lower() for k in ["inbox", "notifications", "alerts", "any updates"]):
            return self._drain_inbox()

        # ── 2. Inline setting change (e.g. "change risk to 2") ─────────────
        setting_response = self._try_setting_change(text)
        if setting_response:
            return setting_response

        # ── 3. Classify intent ─────────────────────────────────────────────
        intent, confidence, entities = self._classify(text)

        # ── 4. Update memory from entities ────────────────────────────────
        self._update_memory(entities)

        # ── 4.5 COGNITIVE: Detect emotion and log user turn ───────────────
        emotion = self._detect_emotion(text)
        self.working_memory.add_turn(ConversationTurn(
            role="user", text=text, intent=intent, emotion=emotion
        ))
        
        # Update symbols discussed in working memory
        if entities.get("symbols"):
            for sym in entities["symbols"]:
                self.working_memory.remember_symbol(sym)

        # ── 5. Handle immediate action intents without agent pipeline ──────
        quick = self._handle_quick_action(intent, text, entities)
        if quick is not None:
            # Log ARIA response to working memory
            self.working_memory.add_turn(ConversationTurn(
                role="aria", text=quick, intent=intent, emotion="neutral"
            ))
            return quick

        # ── 6. Run agentic pipeline ────────────────────────────────────────
        response = self.agent.run(
            intent   = intent,
            entities = entities,
            memory   = self.memory,
            step_callback = self._step_callback,
        )

        # ── 6.5 COGNITIVE: Inner monologue reasoning ───────────────────────
        agent_result = {
            "action": self.agent.last_plan.context.get("action", "WAIT") if self.agent.last_plan else "WAIT",
            "confidence": self.agent.last_plan.context.get("confidence", 0.0) if self.agent.last_plan else 0.0,
        }
        thoughts = self.inner_monologue.think(intent, entities, agent_result)

        # ── 6.7 COGNITIVE: Voice layer naturalization ──────────────────────
        # Transform raw response through the voice layer for natural, varied output
        natural_response = self.voice_layer.render(response, thoughts, intent)

        # ── 7. Post-process: offer to execute if agent recommended action ──
        plan = self.agent.last_plan
        if plan and plan.suggested_action and intent in (
            "execute_trade", "open_buy", "open_sell", "trade_execution"
        ):
            natural_response = self._maybe_execute(natural_response, plan, entities)

        # ── 8. COGNITIVE: Log ARIA turn and update user model ──────────────
        self.working_memory.add_turn(ConversationTurn(
            role="aria", text=natural_response, intent=intent, emotion="neutral"
        ))
        
        # Update user model based on what just happened
        if intent in ("execute_trade", "open_buy", "open_sell"):
            self.user_model.observe("asked_for_execution", {"symbol": self.memory.get("last_symbol")})
        
        # ── 9. Persist memory ──────────────────────────────────────────────
        self.memory["last_intent"] = intent
        return natural_response
    
    def notify(self, msg: str, priority: str = "normal"):
        """Receive system alerts (for compatibility with background scanner)."""
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        if priority in ("critical", "trade_executed"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = "🚨" if priority == "critical" else "✅"
            print(f"\n[{timestamp}] [ARIA]: {prefix} {msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify(self, text: str) -> tuple[str, float, dict]:
        """
        Returns (intent, confidence, entities).
        Keyword shortcuts take precedence over the neural classifier.
        """
        lower = text.lower()
        entities = self.nlp.extract_entities(text)

        # Keyword override
        for keywords, intent in self._KEYWORD_INTENTS:
            if any(k in lower for k in keywords):
                return intent, 1.0, entities

        # Neural classifier
        intent_data = self.nlp.process(text)
        intent      = intent_data.get("intent", "general")
        confidence  = intent_data.get("confidence", 0.0)

        # Symbol mentioned but no specific intent → analyze it
        syms = entities.get("symbols", [])
        if syms and intent not in (
            "execute_trade", "open_buy", "open_sell",
            "close_position", "get_price", "ai_analysis",
        ) and confidence < 0.60:
            intent = "analyze_symbol"

        return intent, confidence, entities

    # ─────────────────────────────────────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────────────────────────────────────

    def _update_memory(self, entities: dict):
        if entities.get("symbols"):
            self.memory["last_symbol"] = entities["symbols"][0]
        if entities.get("timeframes"):
            self.memory["last_timeframe"] = entities["timeframes"][0]

    # ─────────────────────────────────────────────────────────────────────────
    # Emotion Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_emotion(self, text: str) -> str:
        """Simple rule-based emotion detection from user input."""
        lower = text.lower()
        if any(w in lower for w in ["why", "again", "still", "come on", "seriously", "ugh", "argh", "damn"]):
            return "frustrated"
        if any(w in lower for w in ["nice", "great", "yes", "let's go", "perfect", "awesome", "excellent"]):
            return "positive"
        if any(w in lower for w in ["worried", "nervous", "scared", "afraid", "concerned", "anxious"]):
            return "anxious"
        if any(w in lower for w in ["sure", "go ahead", "do it", "confirm", "let's", "ready"]):
            return "confident"
        return "neutral"

    # ─────────────────────────────────────────────────────────────────────────
    # Quick-action handlers (no agent pipeline needed)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_quick_action(self, intent: str, text: str, entities: dict) -> Optional[str]:
        """
        Handle intents that need direct broker calls rather than analysis.
        Returns a response string, or None to fall through to the agent.
        """
        lower = text.lower()

        # Explicit trade execution confirmation
        if intent in ("execute_trade", "open_buy", "open_sell") and self._is_explicit_execute(lower):
            return self._execute_trade_now(entities, intent)

        # Close specific symbol
        if intent == "close_position":
            symbol = (entities.get("symbols") or [self.memory.get("last_symbol")])[0]
            if not symbol:
                self.pending_action = "awaiting_close_symbol"
                return "Which symbol should I close? (or type 'cancel')"
            self.pending_action = "confirm_close"
            self.pending_data   = {"symbol": symbol}
            return (
                f"Close {symbol}? "
                f"Say 'yes' to confirm or 'cancel'."
            )

        # Close profitable positions — MOVED ABOVE close_all to prevent substring match collision
        if intent == "close_profitable_positions" or any(k in lower for k in ["close all profitable", "close profitable", "take profit", "take profits"]):
            # Extract symbol if specified (e.g., "close profitable EURUSD")
            symbols = entities.get("symbols", [])
            symbol = symbols[0] if symbols else None
            
            if symbol:
                # Close profitable positions for a specific symbol
                positions = self.broker.getPositions()
                if not positions:
                    return f"No open positions for {symbol}."
                symbol_positions = [p for p in positions if p.symbol.upper() == symbol.upper()]
                if not symbol_positions:
                    return f"No open positions for {symbol}."
                profitable = [p for p in symbol_positions if p.profit > 0]
                if not profitable:
                    return f"No profitable positions for {symbol}."
                result = self.broker.close_profitable_positions(symbol=symbol)
            else:
                # Close all profitable positions
                result = self.broker.close_profitable_positions()
            
            if isinstance(result, list):
                return "\n".join(result)
            return result

        # Close all confirmation — NOW SAFELY BELOW profitable check
        if intent == "close_all":
            positions = self.broker.getPositions()
            n = len(positions) if positions else 0
            if n == 0:
                return "No open positions to close."
            floating = sum(p.profit for p in positions)
            self.pending_action = "confirm_close_all"
            return (
                f"Close all {n} position(s)? Floating P&L: ${floating:+.2f}. "
                f"Say 'yes' to confirm."
            )

        # Retrain model
        if intent == "retrain_model":
            symbol = self.memory.get("last_symbol")
            if not symbol:
                return "No symbol in memory — mention a symbol first, then ask me to retrain."
            return self.executor.retrain(symbol)

        # Market status queries
        if intent == "market_status":
            return self._handle_market_status()

        # Check notifications
        if intent == "check_notifications":
            guard = self.profit_guard.status()
            inbox = self._drain_inbox()
            return  f"{inbox}\n\n{guard}"

        # Browse symbols
        if intent == "browse_symbols":
            return self._handle_symbol_browser(text, entities)

        # Add symbols to portfolio (comma-separated or joined)
        if intent == "add_symbols" or any(k in lower for k in ["add to portfolio", "add symbols", "add pairs"]):
            return self.add_symbols_to_portfolio(text)

        # Show current portfolio symbols (with smart dynamic grouping and filtering)
        if intent == "trading_symbols" or "portfolio" in lower:
            return self._handle_trading_symbols_intent(lower)
        
        return None  # fall through to agent

    def _handle_market_status(self) -> str:
        """
        Answers 'what can I trade now?' with a session-aware summary.
    
        On a Saturday:
        - Shows crypto as the only open market
        - Lists which portfolio symbols are tradeable vs closed
        - Suggests crypto pairs if the portfolio has none
        - Offers to scan crypto immediately
        """
    
        mgr = MarketSessionManager()
        now_utc = datetime.now(timezone.utc)
    
        
        portfolio_symbols = profile.symbols()
    
        summary = mgr.get_market_status_summary(portfolio_symbols, now_utc)
    
        open_syms, _ = mgr.filter_tradeable_symbols(portfolio_symbols, now_utc)
    
        lines = [summary]
    
        if open_syms:
            lines.append(
                f"\nI can scan {', '.join(open_syms)} right now. "
                f"Say 'scan' to go, or name a symbol to analyse it."
            )
        else:
            # Nothing in portfolio is open — suggest crypto
            suggestions = mgr.suggest_always_open()
            lines.append(
                f"\nYour current portfolio ({', '.join(portfolio_symbols or ['none'])}) "
                f"is all in closed markets right now."
            )
            lines.append(
                f"Crypto trades 24/7. Consider adding: {', '.join(suggestions[:3])}. "
                f"Say 'add BTCUSD to portfolio' to get started."
            )
    
        return "\n".join(lines)
    
    
    def _handle_trading_symbols_intent(self, lower: str) -> str:
        """Return grouped portfolio display. Called from _handle_quick_action."""
        from manager.profile_manager import profile as _p
        from manager.market_sessions import MarketSessionManager
        from collections import defaultdict
    
        syms = _p.symbols()
        if not syms:
            return "Your portfolio is currently empty. Say 'add EURUSD' to get started."
    
        mgr = MarketSessionManager()
    
        # Specific category filter?
        target_category = None
        if any(w in lower for w in ["crypto", "bitcoin", "coin", "digital"]):
            target_category = "crypto"
        elif any(w in lower for w in ["forex", "currencies", "currency", "fx", "fiat"]):
            target_category = "forex"
        elif any(w in lower for w in ["metal", "metals", "gold", "silver"]):
            target_category = "metals"
        elif any(w in lower for w in ["index", "indices"]):
            target_category = "indices"
        elif any(w in lower for w in ["commodity", "commodities", "oil", "energy"]):
            target_category = "commodities"
        elif any(w in lower for w in ["stock", "stocks", "equity", "equities"]):
            target_category = "stocks"
    
        if target_category:
            filtered = [s for s in syms if target_category in mgr.get_symbol_category(s)]
            if not filtered:
                return f"You don't have any {target_category} symbols in your portfolio right now."
            emoji = {"crypto": "₿", "forex": "💱", "metals": "🥇",
                    "indices": "📈", "commodities": "🛢", "stocks": "🏢"}.get(target_category, "📊")
            return f"{emoji} You have {len(filtered)} {target_category} symbol(s): {', '.join(filtered)}"
    
        grouped: defaultdict[str, list] = defaultdict(list)
        for s in syms:
            grouped[mgr.get_symbol_category(s)].append(s)
    
        CAT_EMOJI = {
            "crypto": "₿", "forex": "💱", "metals": "🥇", "commodities": "🛢",
            "stocks": "🏢", "indices_us": "🇺🇸", "indices_eu": "🇪🇺", "indices_asia": "🌏"
        }
        lines = [f"📊 You are currently tracking {len(syms)} symbol(s):"]
        for cat in sorted(grouped):
            items = grouped[cat]
            emoji = CAT_EMOJI.get(cat, "🔹")
            display = cat.replace("_", " ").title()
            lines.append(f"  {emoji} {display} ({len(items)}): {', '.join(items)}")
        return "\n".join(lines)
    
    def _make_session_aware_greeting(self) -> str:
        from manager.market_sessions import MarketSessionManager

        session_manager = MarketSessionManager()
        watchlist = getattr(self.pm, "master_watchlist", [])
        if not watchlist:
            return "Welcome. Your watchlist is empty, so I can't check market status yet."

        open_syms, closed_syms = session_manager.filter_tradeable_symbols(watchlist)
        if not open_syms:
            return (
                "Markets are closed for your watchlist right now. "
                "Crypto is still open 24/7 if you want active exposure."
            )

        if closed_syms:
            return (
                f"Markets are open for {len(open_syms)} of your symbols; "
                f"{len(closed_syms)} are currently closed."
            )

        return "Markets are currently open for all symbols in your watchlist."

    def _is_explicit_execute(self, lower: str) -> bool:
        """Detect strong execute intent (not just analysis)."""
        execute_words = [
            "execute", "place the trade", "place trade", "open trade",
            "go long", "go short", "pull the trigger", "enter now",
            "buy now", "sell now", "trade it", "take the trade",
        ]
        return any(w in lower for w in execute_words)

    def _execute_trade_now(self, entities: dict, intent: str) -> str:
        """Execute a trade based on entities and intent."""
        symbol    = (entities.get("symbols") or [self.memory.get("last_symbol")])[0]
        direction = entities.get("direction")

        if not symbol:
            self.pending_action = "awaiting_trade_symbol"
            self.pending_data   = {"intent": intent}
            return "Which symbol? (e.g. EURUSD)"

        # If direction not in message, use last agent signal
        if not direction:
            last_signal = self.memory.get("last_signal", {})
            direction   = last_signal.get("action", "BUY")
            if direction == "WAIT":
                return (
                    f"My last signal on {symbol} was WAIT — confidence too low. "
                    f"Force a BUY or SELL? Say 'buy {symbol}' or 'sell {symbol}'."
                )

        # NEW: Check last plan grade before executing
        plan = self.agent.last_plan
        if plan:
            scored = next(
                (s.result for s in plan.steps if s.name == "quality_score"), {}
            ) or {}
            grade = scored.get("grade", "D")
            if grade not in ("A", "B"):
                return (
                    f"Signal grade is {grade} — not strong enough to execute. "
                    f"Wait for a Grade A or B setup on {symbol}."
                )

        # Run quick risk check
        cfg = json.loads(self.PROFILE_FILE.read_text()) if self.PROFILE_FILE.exists() else {}
        allowed, reason = self.rm.is_trading_allowed(
            symbol, cfg.get("max_daily_loss", 500.0),
            max(len(cfg.get("trading_symbols", [])), 1)
        )
        if not allowed:
            return f"🛑 Trade blocked: {reason}"

        return self.executor.execute_trade(symbol, direction)

    def _handle_symbol_browser(self, text: str, entities: dict) -> str:
        """
        Searches the broker for available symbols, optionally filtering by
        category or keyword, and returns a grouped, human-readable result.

        Understands:
          "what forex pairs do you have"    → category=forex
          "what gold symbols are available" → query=XAU
          "search USD"                      → query=USD
          "what crypto can I trade"         → category=crypto
          "find GBPJPY"                     → query=GBP
        """
        import re
        from collections import defaultdict

        lower = text.lower()

        # ── Detect category intent ────────────────────────────────────────
        category = None
        if any(w in lower for w in ["forex", "currency", "currencies", "fx", "pair", "pairs"]):
            category = "forex"
        elif any(w in lower for w in ["metal", "metals", "gold", "silver", "xau", "xag", "platinum"]):
            category = "metals"
        elif any(w in lower for w in ["crypto", "bitcoin", "ethereum", "btc", "eth", "digital"]):
            category = "crypto"
        elif any(w in lower for w in ["index", "indices", "stocks", "us30", "nas", "dax", "dow"]):
            category = "indices"
        elif any(w in lower for w in ["oil", "commodity", "commodities", "ngas", "energy"]):
            category = "commodities"

        # ── Detect search keyword ─────────────────────────────────────────
        # Priority: explicit "search X" or "find X" phrase,
        # then NLP-extracted symbols, then currency mentions in text.
        query = None

        search_match = re.search(
            r"(?:search|find|look\s*up|show\s*me|what\s+is\s+the\s+(?:ticker|symbol)\s+for)\s+([A-Za-z]+)",
            text, re.IGNORECASE
        )
        if search_match:
            query = search_match.group(1).upper()

        if not query and entities.get("symbols"):
            # Use first 3 chars of extracted symbol as prefix search
            query = entities["symbols"][0][:3].upper()

        if not query:
            # Fall back to any 3-letter currency code mentioned
            currencies = re.findall(
                r"\b(USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD|SGD|HKD|NOK|SEK|DKK|ZAR|MXN)\b",
                text.upper()
            )
            if currencies:
                query = currencies[0]

        # ── Fetch from broker ─────────────────────────────────────────────
        if not self.broker.connected:
            return "Not connected to MT5 — can't browse symbols. Connect first."

        symbols = self.broker.search_symbols(
            query=query,
            category=category,
            max_results=60,
        )

        if not symbols:
            hint = ""
            if query:
                hint = f" No results for '{query}'."
            if category:
                hint += f" No {category} symbols found."
            return (
                f"Nothing came back from the broker.{hint} "
                f"Try: 'show forex pairs', 'search EURUSD', "
                f"'what metals can I trade', or 'what crypto is available'."
            )

        # ── Format grouped output ─────────────────────────────────────────
        grouped: defaultdict[str, list] = defaultdict(list)
        for s in symbols:
            grouped[s["category"]].append(s)

        header = f"Found {len(symbols)} symbol(s)"
        if query:
            header += f" matching '{query}'"
        if category:
            header += f" in {category}"
        header += " on your broker:\n"

        CAT_EMOJI = {
            "forex":       "💱 FOREX",
            "metals":      "🥇 METALS",
            "crypto":      "₿  CRYPTO",
            "indices":     "📈 INDICES",
            "commodities": "🛢  COMMODITIES",
        }

        lines = [header]
        for cat in ["forex", "metals", "crypto", "indices", "commodities"]:
            items = grouped.get(cat, [])
            if not items:
                continue

            lines.append(f"{CAT_EMOJI.get(cat, cat.upper())} ({len(items)})")

            # Show up to 20 per category to keep the response readable
            for item in items[:20]:
                name   = item["name"]
                desc   = item["description"]
                spread = item["spread_pips"]

                # Only show description when it adds info beyond the ticker name
                desc_str   = f"  {desc}" if desc and desc.upper() != name.upper() else ""
                spread_str = f"  [{spread}p spread]" if spread is not None else ""

                lines.append(f"  {name}{desc_str}{spread_str}")

            if len(items) > 20:
                lines.append(f"  … and {len(items) - 20} more. "
                              f"Narrow it down: 'search EUR' or 'search GBP'.")
            lines.append("")

        lines.append("To add any symbol: 'add EURUSD to portfolio'")
        lines.append("To search further: 'search USD', 'show me crypto', 'find gold symbol'")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-analysis action offer
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_execute(self, response: str, plan, entities: dict) -> str:
        """
        After analysis, if the agent suggests a trade, check if the user
        has implicitly confirmed it (e.g. said 'trade it') or set a pending state.
        """
        suggested = plan.suggested_action  # e.g. "BUY 0.01 EURUSD"
        if not suggested:
            return response

        # Parse suggested action
        parts = suggested.split()
        if len(parts) >= 3:
            direction = parts[0]
            symbol    = parts[2]
            # Cache the last signal action
            self.memory["last_signal"] = {
                "action": direction, "symbol": symbol,
            }
            # Set pending so the user can confirm
            self.pending_action = "confirm_trade"
            self.pending_data   = {"direction": direction, "symbol": symbol,
                                   "lots": float(parts[1]) if len(parts) > 1 else None}
            response += (
                f"\n\nReady to execute {direction} {symbol}? "
                f"Say 'yes' to confirm or 'cancel'."
            )

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-turn pending state machine
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_pending(self, text: str) -> tuple[bool, str]:
        """
        Returns (handled, response).
        True means the pending action consumed this turn.
        """
        lower  = text.lower().strip()
        action = self.pending_action

        # ── Confirm trade ──────────────────────────────────────────────────
        if action == "confirm_trade":
            if lower in ("yes", "y", "go", "do it", "confirm", "execute"):
                d   = self.pending_data
                symbol = d["symbol"]
                
                resp = self.executor.execute_trade(symbol, d["direction"], d.get("lots"))
                self._clear_pending()
                return True, resp
            if lower in ("no", "n", "cancel", "abort", "skip"):
                self._clear_pending()
                return True, "Trade cancelled. What's next?"
            # Not yes/no — let them rephrase
            d = self.pending_data
            return True, (
                f"Waiting for your confirmation on {d['direction']} {d['symbol']}. "
                f"Say 'yes' to execute or 'cancel'."
            )

        # ── Confirm close ──────────────────────────────────────────────────
        if action == "confirm_close":
            if lower in ("yes", "y", "confirm", "close it"):
                symbol = self.pending_data.get("symbol")
                resp   = self.executor.close_position(symbol)
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Close cancelled."

        # ── Confirm close all ──────────────────────────────────────────────
        if action == "confirm_close_all":
            if lower in ("yes", "y", "confirm", "close all"):
                resp = self.executor.close_all()
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Close-all cancelled."

        # ── Awaiting trade symbol ──────────────────────────────────────────
        if action == "awaiting_trade_symbol":
            entities = self.nlp.extract_entities(text)
            symbols  = entities.get("symbols", [])
            if symbols:
                symbol = symbols[0]
                self.memory["last_symbol"] = symbol
                intent = self.pending_data.get("intent", "execute_trade")
                self._clear_pending()
                return True, self._execute_trade_now(entities, intent)
            if lower == "cancel":
                self._clear_pending()
                return True, "Trade cancelled."
            return True, "Still need a symbol — try 'EURUSD' or type 'cancel'."

        # ── Awaiting close symbol ──────────────────────────────────────────
        if action == "awaiting_close_symbol":
            entities = self.nlp.extract_entities(text)
            symbols  = entities.get("symbols", [])
            if symbols:
                symbol = symbols[0]
                self._clear_pending()
                return True, self.executor.close_position(symbol)
            if lower == "cancel":
                self._clear_pending()
                return True, "Close cancelled."
            return True, "Which symbol to close? (or 'cancel')"

        # ── Retry micro-lot ────────────────────────────────────────────────
        if action == "retry_micro_lot":
            if lower in ("yes", "y"):
                d = self.pending_data
                symbol = d["symbol"]
                
                resp = self.executor.execute_trade(symbol, d["direction"], lots=0.01)
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Retry skipped."

        return False, ""

    def _clear_pending(self):
        self.pending_action = None
        self.pending_data   = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup and Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Gracefully shutdown ARIA and all background processes."""
        try:
            self.episodic_memory.flush()
            self.proactive_engine.stop()
            self.trailing_manager.stop()
            self.profit_guard.stop()
            self.message_processor.shutdown(wait=False)
        except Exception as e:
            print(f"Error during shutdown: {e}")


    # ─────────────────────────────────────────────────────────────────────────
    # Trade Cooldown Management
    # ─────────────────────────────────────────────────────────────────────────

    def set_cooldown_duration(self, minutes: int):
        """Set the cooldown period (in minutes) after closing a trade."""
        seconds = max(0, int(minutes * 60))
        try:
            self.broker.set_cooldown(seconds)
            if seconds > 0:
                self._type_print(f"Cooldown set to {minutes} minute(s). "
                               f"Cannot re-trade a symbol for {minutes}min after closing it.")
            else:
                self._type_print("Cooldown disabled.")
        except Exception:
            self._type_print("Failed to update broker cooldown setting.")


    # ─────────────────────────────────────────────────────────────────────────
    # Setting change handler
    # ─────────────────────────────────────────────────────────────────────────

    _SETTING_MAP = {
        "daily loss":       ("max_daily_loss",  "Max Daily Loss"),
        "daily loss limit": ("max_daily_loss",  "Max Daily Loss"),
        "max loss":         ("max_daily_loss",  "Max Daily Loss"),
        "risk":             ("risk_percentage", "Risk Per Trade"),
        "risk percentage":  ("risk_percentage", "Risk Per Trade"),
        "risk per trade":   ("risk_percentage", "Risk Per Trade"),
        "target":           ("target_profit",   "Target Profit"),
        "target profit":    ("target_profit",   "Target Profit"),
        "stop loss":        ("stop_loss",        "Stop Loss"),
        "sl":               ("stop_loss",        "Stop Loss"),
        "cooldown":         ("cooldown_duration_minutes", "Cooldown Duration"),
        "lock balance":     ("lock_balance",     "Lock Balance"),
        "lock":             ("lock_balance",     "Lock Balance"),
        "lock pct":         ("lock_balance_pct", "Lock Balance %"),
    }

    def _try_setting_change(self, text: str) -> Optional[str]:
        match = re.search(r"change (?:my )?(.*?)\s+to\s+([0-9.]+)", text, re.IGNORECASE)
        follow = re.search(r"(?:make|set|change) it (?:to\s*)?([0-9.]+)", text, re.IGNORECASE)
        follow_key = re.search(r"no,?\s*(?:make|set) (?:it\s*)?([0-9.]+)", text, re.IGNORECASE)

        target_key = new_value = None

        if follow and self.memory.get("last_setting"):
            target_key = self.memory["last_setting"]
            new_value  = float(follow.group(1))
        if follow_key and self.memory.get("last_setting"):
            target_key = self.memory["last_setting"]
            new_value  = float(follow_key.group(1))
        if match:
            target_key = match.group(1).strip().lower()
            new_value  = float(match.group(2))

        if target_key and new_value is not None:
            if target_key in self._SETTING_MAP:
                cfg_key, display = self._SETTING_MAP[target_key]
                
                # Handle cooldown specially (not in config file)
                if cfg_key == "cooldown_duration_minutes":
                    self.set_cooldown_duration(int(new_value))
                    self.memory["last_setting"] = target_key
                    return f"Updated {display} to {int(new_value)} minute(s). Saved."
                
                # Handle other config settings
                self._update_config(cfg_key, new_value)
                self.memory["last_setting"] = target_key
                return f"Updated {display} to {new_value}. Saved."
            return (
                f"I don't recognise '{target_key}'. "
                f"Valid settings: risk, target, stop loss, daily loss, cooldown."
            )
        return None

    def _update_config(self, key: str, value: float):
        # Map old flat keys to new structure
        _RISK_KEYS = {
            "risk_percentage":  "risk_pct",
            "stop_loss":        "stop_loss_pips",
            "target_profit":    "take_profit_pips",
            "max_daily_loss":   "max_daily_loss",
            "daily_goal":       "daily_goal",
            "lock_balance":     "lock_amount",
            "lock_balance_pct": "lock_pct",
        }
        _BROKER_KEYS = {
            "cooldown_duration_minutes": None,   # handled separately via broker
        }
        if key in _RISK_KEYS:
            profile.update_risk(**{_RISK_KEYS[key]: value})
            # Keep LockBalanceGuard in sync
            r = profile.risk()
            self.rm.lock_guard.update(lock_amount=r.lock_amount, lock_pct=r.lock_pct_decimal)
        elif key == "cooldown_duration_minutes":
            self.broker.set_cooldown(int(value * 60))
        
    def add_symbols_to_portfolio(self, text: str) -> str:
        """
        Parse comma-separated or 'and'-joined symbols from user text and add them to profile.
        
        Examples:
          "add USD, EUR and JPY"           → extracts USD, EUR, JPY
          "add EURUSD, USDJPY and GBPUSD" → adds all three
          "add USDCHF USDCAD USDSEK"      → adds all three
        """
        import re
        from manager.nlp_engine import SYMBOL_ALIASES
        
        # Extract all potential symbols (6-7 letter uppercase words or 3-letter codes)
        symbols_found = re.findall(r"\b([A-Z]{3,7})\b", text.upper())
        
        # Resolve aliases
        resolved = set()
        for sym in symbols_found:
            # Check if it's an alias (like USD → EURUSD, or a 3-letter code like USD)
            if sym in SYMBOL_ALIASES:
                resolved.add(SYMBOL_ALIASES[sym])
            elif len(sym) >= 6:
                # Likely a direct symbol like EURUSD
                resolved.add(sym)
            elif len(sym) == 3:
                # Single currency code—try common pairs
                # USD → EURUSD, GBP → GBPUSD, JPY → USDJPY, etc.
                if sym == "USD":
                    # User said "USD" — add common USD pairs
                    resolved.update(["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD"])
                elif sym == "EUR":
                    resolved.update(["EURUSD", "EURGBP", "EURCHF", "EURJPY"])
                elif sym == "GBP":
                    resolved.update(["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF"])
                elif sym == "JPY":
                    resolved.update(["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"])
                elif sym in SYMBOL_ALIASES:
                    resolved.add(SYMBOL_ALIASES[sym])
        
        if not resolved:
            return "❌ No symbols recognized. Try: 'add EURUSD, USDJPY' or 'add USD, EUR'."
        
        
        
        # Get existing symbols
        existing = set(profile.symbols())
        added = set()
        for sym in resolved:
            if profile.add_symbol(sym):
                added.add(sym)
        if not added:
            return f"ℹ️  All symbols already in portfolio: {', '.join(sorted(existing))}"
        
        updated = profile.symbols()
        return (
            f"✅ Added {len(added)} symbol(s): {', '.join(sorted(added))}\n"
            f"📊 Portfolio now tracks ({len(updated)}): {', '.join(updated)}"
        )
       

    # ─────────────────────────────────────────────────────────────────────────
    # Notification inbox
    # ─────────────────────────────────────────────────────────────────────────

    def _drain_inbox(self) -> str:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Agent step callback  (progress indicator)
    # ─────────────────────────────────────────────────────────────────────────

    def _step_callback(self, name: str, description: str):
        """Print a dim reasoning step during agent execution."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [ARIA ▸]: {description}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Startup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _broker_login(self):
        """Try saved credentials, then prompt interactively."""
        if self.broker.connected:
            return
        
        data = self.profile_manager.load_credentials()
        login, password, server = (
            data.get("login"), data.get("password"), data.get("server")
        )

        if login and password and server:
            if self.broker.connect(login=login, password=password, server=server):
                return
            self._type_print("Saved credentials failed — please log in again.")

        while not self.broker.connected:
            try:
                login = int(input("[ARIA]: MT5 Account Number: ").strip())
            except ValueError:
                self._type_print("Account number must be an integer.")
                continue
            if len(str(login)) < 5:
                self._type_print("Account number too short.")
                continue

            password = getpass.getpass("[ARIA]: Password: ")
            if not password:
                self._type_print("Password can't be empty.")
                continue

            server = input("[ARIA]: Broker Server (e.g. MetaQuotes-Demo): ").strip()
            if not server:
                self._type_print("Server can't be empty.")
                continue

            if self.broker.connect(login=login, password=password, server=server):
                self.profile_manager.save_credentials(login, password, server)
                self._type_print(f"Connected. Account #{self.broker.login} is live.")
            else:
                self._type_print("Connection failed — verify credentials and check MT5 is running.")

    def _load_or_setup_config(self):
        if self.PROFILE_FILE.exists():
            try:
                raw = json.loads(self.PROFILE_FILE.read_text())
                if "portfolio" in raw and raw["portfolio"].get("symbols"):
                    return  # new structured format already in place — nothing to do
                
            except Exception:
                pass
            

    def _setup_config(self):
        """Interactive first-time setup."""
        print("\n" + "=" * 45 + "\n🤖  ARIA FIRST-TIME SETUP\n" + "=" * 45)
        user_input = input(
            "[ARIA]: Describe your goal "
            "(e.g. 'Risk 1%, make $20 daily on EURUSD, GBPUSD'): "
        ).strip()

        text_upper = user_input.upper()
        symbols = list(set(re.findall(r"\b[A-Z]{6}\b", text_upper)))
        risk_m  = re.search(r"(\d+(?:\.\d+)?)\s*%", user_input)
        profit_m= re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:daily|per day)?", user_input)

        symbols       = symbols or ["EURUSD"]
        target_profit = round(daily_goal / max(len(symbols), 1), 2)
        
        risk_pct   = float(risk_m.group(1))   if risk_m   else 1.0
        daily_goal = float(profit_m.group(1)) if profit_m else 10.0
        tp_pips    = round(daily_goal / max(len(symbols), 1), 2)
    

        cfg = {
        "_version": 2,
        "_note": "Edit values here. Do NOT rename top-level sections.",
        "portfolio": {
            "symbols": symbols,
            "preferred_timeframes": ["H1"],
            "asset_classes": {},
            "strategy_mapping": {
                "default": "Mean_Reversion",
                "asset_class_defaults": {
                    "forex":  "Mean_Reversion",
                    "metals": "Momentum",
                    "crypto": "Momentum",
                    "indices": "Breakout",
                },
                "symbol_overrides": {}
            }
        },
        "risk": {
            "defaults": {
                "risk_pct":         risk_pct,
                "stop_loss_pips":   20.0,
                "take_profit_pips": tp_pips,
                "max_daily_loss":   daily_goal * 2,
                "daily_goal":       daily_goal,
                "cooldown_minutes": 5,
                "lock_amount":      0.0,
                "lock_pct":         0.0,
            },
            "symbol_overrides": {},
            "drawdown_recovery": {
                "moderate_threshold": 0.5,
                "moderate_scale":     0.5,
                "critical_threshold": 0.8,
                "critical_scale":     0.25
            }
        },
        "broker": {
            "max_open_trades":       3,
            "min_margin_level":      150.0,
            "spread_tolerance_pips": 3.0,
            "magic_number":          1000,
            "slippage_points":       20
        },
        "sessions": {
            "avoid_asian_session":  True,
            "avoid_friday_close":   True,
            "asian_session_pairs":  ["USDJPY", "AUDUSD", "NZDUSD", "CADJPY", "CHFJPY", "GBPJPY"]
        },
        "scanner": {
            "interval_seconds":     3,
            "dry_run":              False,
            "mtf_min_alignment":    0.50,
            "volatility_spike_atr": 2.5,
            "dead_volume_ratio":    0.3
        }
    }
        self.PROFILE_FILE.parent.mkdir(exist_ok=True)
        self.PROFILE_FILE.write_text(json.dumps(cfg, indent=4))
        self._type_print(
            f"Config saved — tracking {len(symbols)} symbol(s), "
            f"{risk_pct}% risk, ${target_profit} target/symbol."
        )

    def _print_banner(self):
        syms = profile.symbols()
        r    = profile.risk()
        print("\n" + "=" * 60)
        print(
            f"🤖  ARIA ONLINE | Risk: {r.risk_pct}% | "
            f"TP: ${r.take_profit_pips} | SL: {r.stop_loss_pips} pips"
        )
        if syms:
            print(f"📊  Tracking ({len(syms)}): {', '.join(syms)}")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Main chat loop
    # ─────────────────────────────────────────────────────────────────────────

    def _run_loop(self):
        try:
            while True:
                with patch_stdout():
                    try:
                        inp = self.session.prompt("[You]: ").strip()
                    except EOFError:
                        self._type_print("Session ended.")
                        break

                if not inp:
                    continue

                lower = inp.lower()
                if lower in ("quit", "exit", "bye"):
                    self._type_print("Logging out. Goodbye.")
                    break

                try:
                    response = self.process_message(inp)
                except Exception as exc:
                    response = f"Something went wrong: {exc}. Try again."

                if response:
                    self._type_print(response)

        except KeyboardInterrupt:
            self._type_print("Interrupt received — shutting down.")

    # ─────────────────────────────────────────────────────────────────────────
    # Output helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _type_print(self, msg: str, delay: float = 0.010) -> None:
        ts        = datetime.now().strftime("%H:%M:%S")
        clean     = msg.replace("[Bot]: ", "").replace("[ARIA]: ", "").strip()
        prefix    = f"[{ts}] [ARIA]: "

        if sys.stdout.isatty():
            # Print prefix instantly, then typewriter for the message
            sys.stdout.write(prefix)
            for char in clean:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay)
            sys.stdout.write("\n")
        else:
            print(f"{prefix}{clean}")

    # ── Legacy compatibility shims ────────────────────────────────────────────

    def bot_print(self, msg: str):
        self._type_print(msg)

    def chat(self, user_message: str, action_result: str = None) -> str:
        if action_result:
            return f"Done — {action_result}"
        return self.process_message(user_message)