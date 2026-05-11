import json
import re
import sys
import time
import getpass
import random
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
from manager.conversational_parser import ConversationalParser
from strategies.strategy_manager import StrategyManager
from trader import Trader
from manager.profile_manager import profile
from manager.position_monitor import PositionMonitor

# ─────────────────────────────────────────────────────────────────────────────
# ActionExecutor — handles broker operations recommended by the agent
# ─────────────────────────────────────────────────────────────────────────────

class ActionExecutor:
    """
    Performs real broker actions after the agent recommends them.
    """

    def __init__(self, broker: Trader, strategy_manager: StrategyManager,
                 risk_manager: RiskManager, profile_path: Path):
        self.broker    = broker
        self.sm        = strategy_manager
        self.rm        = risk_manager
        self.profile   = profile_path

    def _execute_trade_action(self, plan: AgentPlan):
        symbol = plan.symbol
        direction = plan.context.get("action", "BUY").upper()
        lot_size = self.rm.calculate_position_size(symbol)
        dynamic_targets = plan.context.get("dynamic_targets", {})
        b = profile.broker()
        s = profile.sessions()
        gk = TradeGatekeeper(
            max_spread_pips       = b.spread_tolerance_pips,
            avoid_asian_session   = s.avoid_asian_session,
            avoid_friday_close    = s.avoid_friday_close,
        )
        ok, reason = gk.gate(symbol, None)
        if not ok:
            return f"Can't place the trade right now — {reason}"

        if direction == "BUY":
            sl_pips = dynamic_targets.get("sl_buy_pips", 50)
            tp_pips = dynamic_targets.get("tp_buy_pips", 100)
        elif direction == "SELL":
            sl_pips = dynamic_targets.get("sl_sell_pips", 50)
            tp_pips = dynamic_targets.get("tp_sell_pips", 100)
        else:
            return "Not sure of the direction — say 'buy' or 'sell' and I'll execute."

        try:
            result = self.broker.execute_trade(
                symbol=symbol,
                action=direction,
                lots=lot_size,
                stop_loss_pips=sl_pips,
                take_profit_pips=tp_pips,
            )
            if result.get("success"):
                return f"Done — {direction} on {symbol} is live (ticket #{result['ticket']})."
            return f"Broker rejected the {direction} on {symbol}: {result.get('reason', 'unknown reason')}."
        except Exception as e:
            return f"Something went wrong during execution: {str(e)}"

    def execute_trade(self, symbol: str, direction: str, lots: Optional[float] = None, strategy: str = "Manual") -> str:
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            return f"'{direction}' isn't a valid direction — say BUY or SELL."
        r = profile.risk()

        if not lots:
            plan = self.rm.calculate_safe_trade(
                symbol         = symbol,
                base_risk_pct  = r.risk_pct,
                stop_loss_pips = r.stop_loss_pips,
                max_daily_loss = r.max_daily_loss,
                portfolio_size = max(len(profile.symbols()), 1),
            )
            if not plan.get("approved"):
                return f"Can't place that trade — {plan.get('reason', 'risk check failed')}."
            lots = plan.get("lots", 0.01)

        result = self.broker.execute_trade(
            symbol           = symbol,
            action           = direction,
            lots             = lots,
            stop_loss_pips   = r.stop_loss_pips,
            take_profit_pips = r.take_profit_pips,
            strategy         = strategy,
        )

        if result.get("success"):
            return (
                f"Done — {direction} {lots} lots of {symbol} is live. "
                f"Ticket #{result['ticket']} filled at {result['price']}."
            )
        return f"Broker rejected that order: {result.get('reason', 'unknown reason')}."

    def close_position(self, symbol: str) -> str:
        success = self.broker.close_position(symbol)
        return f"Closed {symbol}." if success else f"Couldn't close {symbol} — check if it's still open."

    def close_all(self) -> str:
        self.broker.close_all_positions()
        return "All positions closed."

    def retrain(self, symbol: str) -> str:
        try:
            self.sm.continuous_learning_routine(symbol)
            return f"Model retrained for {symbol} — should be sharper next signal."
        except Exception as e:
            return f"Retraining ran into an issue: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# ARIA  —  main chatbot class
# ─────────────────────────────────────────────────────────────────────────────

class ARIA:
    """
    Adaptive Reasoning & Intelligence for Algo-trading.

    Conversational trading partner — not a command-line tool.
    """

    PROFILE_FILE = Path("data/profile.json")

    # ── Minimal keyword fallbacks (last resort only) ──────────────────────────
    _HARD_KEYWORDS: list[tuple[list[str], str]] = [
        (["inbox", "notifications", "alerts check", "any updates"], "check_notifications"),
        (["retrain", "update model", "relearn"], "retrain_model"),
        (["browse symbols", "search symbols", "available symbols", "list all symbols"], "browse_symbols"),
        (["add to portfolio", "add symbol", "add pair"], "add_symbols"),
    ]

    def __init__(
        self,
        intents_filepath: str,
        broker: Trader,
        strategy_manager: StrategyManager,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
    ):
        self.broker    = broker
        self.sm        = strategy_manager
        self.pm        = portfolio_manager
        self.rm        = risk_manager

        self.nlp = NLPEngine(
            intents_filepath=intents_filepath,
            data_dir=str(Path("data"))
        )

        self.profile_manager = ProfileManager(path=Path("data/profile.json"))
        self.reasoning = ReasoningEngine(strategy_manager, risk_manager, portfolio_manager)
        self.responder = ResponseEngine()
        self.agent = AgentCore(strategy_manager, risk_manager, portfolio_manager,
                               broker, self.reasoning)
        self.executor = ActionExecutor(broker, strategy_manager, risk_manager, self.PROFILE_FILE)

        self.dynamic_targeter = DynamicRiskTargeter(self.broker)
        self.trailing_manager = TrailingStopManager(self.broker, self.dynamic_targeter)
        self.trailing_manager.start()
        self.profit_guard = ProfitGuard(self.broker, notify_callback=self.receive_system_alert)
        self.profit_guard.start()

        self.memory: dict[str, Any] = {
            "last_symbol":    None,
            "last_signal":    None,
            "last_intent":    None,
            "last_timeframe": "H1",
            "last_setting":   None,
        }

        # ── Cognitive components ──────────────────────────────────────────────
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

        # ── Conversational parser ─────────────────────────────────────────────
        self.conv_parser = ConversationalParser()

        self.proactive_engine.start()
        self.user_model.observe("session_start")
        self.position_monitor = PositionMonitor(
            broker=self.broker,
            on_external_close=self._on_external_close,
        )
        self.broker.register_position_monitor(self.position_monitor)
        self.position_monitor.start()

        # Multi-turn state
        self.pending_action: Optional[str] = None
        self.pending_data: dict = {}

        # Notification inbox
        self.notification_inbox: list = []
        self.inbox_lock = threading.Lock()

        # Message processing timeout
        self.message_processor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ARIA-msg")
        self.message_timeout_seconds = 30

        self.console = Console()
        self.session = PromptSession()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    # chat.py — add this method to ARIA class

    def _on_external_close(
        self,
        ticket: int,
        symbol: str,
        profit: float,
        close_price: float,
        direction: str,
        lots: float,
        open_price: float,
    ) -> None:
        """
        Fired when a position closes outside the bot — SL/TP hit,
        manual close from the MT5 app, margin call, etc.
        Keeps all internal state consistent.
        """
        sign   = "+" if profit >= 0 else ""
        reason = "TP hit" if profit > 0 else "SL hit" if profit < 0 else "break-even close"

        # 1. Notify the user
        self.receive_system_alert(
            f"📱 External close — {symbol} {direction} {lots}L "
            f"@ {close_price} ({reason}): {sign}${profit:.2f}",
            priority="trade_executed",
        )

        # 2. Log to trade history CSV (same format as bot closes)
        strategy = self.broker._strategy_for(ticket)
        self.broker._log_trade_history(
            action="CLOSE",
            symbol=symbol,
            lots=lots,
            price=close_price,
            ticket=ticket,
            comment=f"External close — {reason}",
            strategy=strategy,
            profit=profit,
        )

        # 3. Apply cooldown so the scanner doesn't immediately re-enter
        self.broker._mark_cooldown(symbol)

        # 4. Update loss-streak tracker (from previous win-rate fix)
        if hasattr(self.rm, "record_win") and hasattr(self.rm, "record_loss"):
            if profit >= 0:
                self.rm.record_win(symbol)
            else:
                self.rm.record_loss(symbol)

        # 5. Feed ML learning pipeline with the outcome
        self.pm.log_trade_for_learning(ticket=ticket, profit=profit)

        # 6. Episodic memory — remember this for future context
        self.episodic_memory.store(Episode(
            timestamp=datetime.now().isoformat(),
            episode_type="trade",
            symbol=symbol,
            summary=f"{symbol} {direction} closed externally ({reason})",
            outcome=f"{sign}${profit:.2f}",
            emotional_tag=None,
            tags=["external_close", "sl_hit" if profit < 0 else "tp_hit", symbol],
        ))

        # 7. Update working memory so ARIA knows what just happened
        self.working_memory.remember_symbol(symbol)
    def receive_system_alert(self, msg: str, priority: str = "normal"):
        with self.inbox_lock:
            self.notification_inbox.append({"msg": msg, "priority": priority})

        if priority in ("critical", "trade_executed"):
            ts   = datetime.now().strftime("%H:%M:%S")
            icon = "🚨" if priority == "critical" else "✅"
            print(f"\n[{ts}] [ARIA]: {icon} {msg}")

    def start_chat(self):
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
        try:
            future = self.message_processor.submit(self._process_message_impl, user_input)
            return future.result(timeout=self.message_timeout_seconds)
        except FuturesTimeoutError:
            mgr = MarketSessionManager()
            portfolio_symbols = profile.symbols()
            open_syms, closed_syms = mgr.filter_tradeable_symbols(portfolio_symbols)

            if closed_syms and not open_syms:
                msg = (
                    f"That request timed out — your portfolio "
                    f"({', '.join(closed_syms[:3])}…) is in closed markets right now, "
                    f"which can hang the broker connection. Try when markets open."
                )
            else:
                msg = "Request timed out — broker might be unresponsive. Try again in a moment."

            self.notify(msg, priority="critical")
            return msg

    def _process_message_impl(self, user_input: str) -> str:
        """
        Conversational pipeline:
        1. Pending state check (multi-turn confirmation)
        2. Conversational parser (natural language → intent)
        3. Hard keyword fallbacks (very specific commands)
        4. NLP classifier (trained neural model)
        5. Agent reasoning + cognitive layers
        """
        text = user_input.strip()
        if not text:
            return ""

        # ── 0. Detect emotion early (shapes everything downstream) ────────────
        emotion = self._detect_emotion(text)

        # ── 1. Log user turn to working memory ───────────────────────────────
        # (Intent filled in after classification)
        self.working_memory.add_turn(ConversationTurn(
            role="user", text=text, intent="", emotion=emotion
        ))

        # ── 2. Pending multi-turn state takes priority ────────────────────────
        if self.pending_action:
            handled, response = self._handle_pending(text)
            if handled:
                self._log_aria_turn(response, self.memory.get("last_intent", ""))
                return response

        # ── 3. Conversational parser (natural language first) ─────────────────
        parsed = self.conv_parser.parse(text, self.working_memory)

        if parsed:
            # Special confirm/cancel handled conversationally
            if parsed.intent == "__confirm__" and self.pending_action:
                handled, response = self._handle_pending("yes")
                if handled:
                    self._log_aria_turn(response, "confirm")
                    return response

            if parsed.intent == "__cancel__" and self.pending_action:
                self._clear_pending()
                response = random.choice([
                    "Sure, leaving it. What else?",
                    "No problem, cancelled.",
                    "Got it, moving on. What do you want to do?",
                ])
                self._log_aria_turn(response, "cancel")
                return response

            # Update entities from conversational parse
            entities = {
                "symbols": parsed.symbols,
                "direction": parsed.direction,
                "timeframes": [],
                "money": [],
                "lots": None,
                "percentages": [],
                "sentiment": "neutral",
            }
            # Merge with NLP entities for symbol coverage
            nlp_entities = self.nlp.extract_entities(text)
            if nlp_entities.get("symbols") and not parsed.symbols:
                entities["symbols"] = nlp_entities["symbols"]
            if nlp_entities.get("lots"):
                entities["lots"] = nlp_entities["lots"]

            intent = parsed.intent
            confidence = parsed.confidence

            # Note if we used context to fill gaps — show it naturally
            context_note = ""
            if parsed.context_used and self.working_memory.last_symbol:
                context_note = f"(using {self.working_memory.last_symbol} from earlier)"

        else:
            # ── 4. Hard keyword fallbacks ─────────────────────────────────────
            lower = text.lower()
            keyword_intent = None
            for keywords, kw_intent in self._HARD_KEYWORDS:
                if any(k in lower for k in keywords):
                    keyword_intent = kw_intent
                    break

            entities = self.nlp.extract_entities(text)

            if keyword_intent:
                intent = keyword_intent
                confidence = 1.0
            else:
                # ── 5. NLP classifier ─────────────────────────────────────────
                intent_data = self.nlp.process(text)
                intent = intent_data.get("intent", "general")
                confidence = intent_data.get("confidence", 0.0)

                # Symbol mentioned but low confidence → default to analyze
                syms = entities.get("symbols", [])
                if syms and confidence < 0.60 and intent not in (
                    "execute_trade", "open_buy", "open_sell",
                    "close_position", "get_price", "ai_analysis",
                ):
                    intent = "analyze_symbol"

            context_note = ""

        # ── 6. Update short-term memory ───────────────────────────────────────
        self._update_memory(entities)
        if entities.get("symbols"):
            for sym in entities["symbols"]:
                self.working_memory.remember_symbol(sym)

        # Update the turn we logged with the resolved intent
        if self.working_memory.turns:
            self.working_memory.turns[-1].intent = intent

        # ── 7. Inline setting change ──────────────────────────────────────────
        setting_response = self._try_setting_change(text)
        if setting_response:
            self._log_aria_turn(setting_response, intent)
            return setting_response

        # ── 8. Direct action handlers (no agent pipeline needed) ──────────────
        quick = self._handle_quick_action(intent, text, entities)
        if quick is not None:
            self._log_aria_turn(quick, intent)
            self.memory["last_intent"] = intent
            return quick

        # ── 9. Agent pipeline ─────────────────────────────────────────────────
        response = self.agent.run(
            intent   = intent,
            entities = entities,
            memory   = self.memory,
            step_callback = self._step_callback,
        )

        # ── 10. Cognitive layers: inner monologue → voice naturalizer ─────────
        agent_result = {
            "action":     self.agent.last_plan.context.get("action", "WAIT") if self.agent.last_plan else "WAIT",
            "confidence": self.agent.last_plan.context.get("confidence", 0.0) if self.agent.last_plan else 0.0,
        }
        thoughts = self.inner_monologue.think(intent, entities, agent_result)
        natural_response = self.voice_layer.render(response, thoughts, intent)

        # Prepend context note naturally if we used memory to fill gaps
        if context_note:
            natural_response = f"{context_note} — {natural_response}"

        # ── 11. Offer to execute if agent recommended a trade ─────────────────
        plan = self.agent.last_plan
        if plan and plan.suggested_action and intent in (
            "execute_trade", "open_buy", "open_sell", "trade_execution"
        ):
            natural_response = self._maybe_execute(natural_response, plan, entities)

        # ── 12. Update cognitive state ────────────────────────────────────────
        self._log_aria_turn(natural_response, intent)
        if intent in ("execute_trade", "open_buy", "open_sell"):
            self.user_model.observe("asked_for_execution", {"symbol": self.memory.get("last_symbol")})
        self.memory["last_intent"] = intent

        return natural_response

    def _log_aria_turn(self, text: str, intent: str):
        """Log ARIA's response to working memory."""
        self.working_memory.add_turn(ConversationTurn(
            role="aria", text=text, intent=intent, emotion="neutral"
        ))

    def notify(self, msg: str, priority: str = "normal"):
        return self.receive_system_alert(msg, priority)

    # ─────────────────────────────────────────────────────────────────────────
    # Emotion Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_emotion(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["why", "again", "still", "come on", "seriously", "ugh", "argh", "damn", "frustrated", "annoying"]):
            return "frustrated"
        if any(w in lower for w in ["nice", "great", "yes", "let's go", "perfect", "awesome", "excellent", "love it"]):
            return "positive"
        if any(w in lower for w in ["worried", "nervous", "scared", "afraid", "concerned", "anxious", "stressed"]):
            return "anxious"
        if any(w in lower for w in ["sure", "go ahead", "do it", "confirm", "let's", "ready", "confident"]):
            return "confident"
        return "neutral"

    # ─────────────────────────────────────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────────────────────────────────────

    def _update_memory(self, entities: dict):
        if entities.get("symbols"):
            self.memory["last_symbol"] = entities["symbols"][0]
        if entities.get("timeframes"):
            self.memory["last_timeframe"] = entities["timeframes"][0]

    # ─────────────────────────────────────────────────────────────────────────
    # Quick-action handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_quick_action(self, intent: str, text: str, entities: dict) -> Optional[str]:
        lower = text.lower()

        # Backtesting
        if "backtest" in lower or "back test" in lower or "test the strategy" in lower:
            return self._handle_backtest(text, entities)

        # Explicit trade execution
        if intent in ("execute_trade", "open_buy", "open_sell") and self._is_explicit_execute(lower):
            return self._execute_trade_now(entities, intent)

        # Close specific symbol
        if intent == "close_position":
            symbol = (entities.get("symbols") or [self.memory.get("last_symbol")])[0]
            if not symbol:
                self.pending_action = "awaiting_close_symbol"
                return "Which position did you want to close?"
            self.pending_action = "confirm_close"
            self.pending_data   = {"symbol": symbol}
            positions = self.broker.getPositions() or []
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos:
                sign = "+" if pos.profit >= 0 else ""
                return (
                    f"You've got {symbol} {('long' if pos.type == 0 else 'short')} "
                    f"at {pos.price_open}, currently {sign}${pos.profit:.2f}. "
                    f"Close it?"
                )
            return f"Close {symbol}? Say yes to confirm."

        # Close profitable positions
        if intent == "close_profitable_positions" or any(k in lower for k in ["close all profitable", "close profitable", "take profit", "take profits"]):
            symbols = entities.get("symbols", [])
            symbol = symbols[0] if symbols else None
            if symbol:
                positions = self.broker.getPositions()
                if not positions:
                    return f"No open positions for {symbol}."
                symbol_positions = [p for p in positions if p.symbol.upper() == symbol.upper()]
                if not symbol_positions:
                    return f"Nothing open on {symbol} right now."
                profitable = [p for p in symbol_positions if p.profit > 0]
                if not profitable:
                    return f"{symbol} is not in profit at the moment."
                result = self.broker.close_profitable_positions(symbol=symbol)
            else:
                result = self.broker.close_profitable_positions()

    def _handle_backtest(self, text: str, entities: dict) -> str:
        import re
        from manager.backtester import run_backtest

        symbol = (entities.get("symbols") or [self.memory.get("last_symbol")])[0]
        if not symbol:
            return "Which symbol do you want to backtest? Mention it and I'll run the numbers."

        # Parse optional date range from text:  "backtest EURUSD from Jan to June"
        start, end = "", ""
        yr_match = re.search(r"20\d{2}", text)
        year = yr_match.group(0) if yr_match else "2024"

        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        months_found = [v for k, v in month_map.items() if k in text.lower()]
        if len(months_found) >= 2:
            start = f"{year}-{months_found[0]}-01"
            end   = f"{year}-{months_found[1]}-28"

        risk_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        risk_pct = float(risk_match.group(1)) if risk_match else 1.0

        self._type_print(
            f"Running backtest on {symbol} "
            f"({'full history' if not start else f'{start} → {end}'}) "
            f"at {risk_pct}% risk per trade..."
        )

        try:
            result = run_backtest(
                strategy_manager = self.sm,
                symbol           = symbol,
                start_date       = start,
                end_date         = end,
                risk_pct         = risk_pct,
                notify_callback  = lambda m, **_: self._step_callback("backtest", m),
            )
            result.save_csv(f"data/backtest_{symbol}.csv")
            return result.summary()
        except Exception as exc:
            return f"Backtest ran into a problem: {exc}"
            if isinstance(result, list):
                return "\n".join(result)
            return result

        # Close all
        if intent == "close_all":
            positions = self.broker.getPositions()
            n = len(positions) if positions else 0
            if n == 0:
                return "Nothing open to close right now."
            floating = sum(p.profit for p in positions)
            sign = "+" if floating >= 0 else ""
            self.pending_action = "confirm_close_all"
            return (
                f"You have {n} position(s) open, floating {sign}${floating:.2f}. "
                f"Close everything?"
            )

        # Retrain
        if intent == "retrain_model":
            symbol = self.memory.get("last_symbol")
            if not symbol:
                return "Which symbol should I retrain the model on? Mention it first."
            return self.executor.retrain(symbol)

        # Market status
        if intent == "market_status":
            return self._handle_market_status()

        # Check notifications
        if intent == "check_notifications":
            guard = self.profit_guard.status()
            inbox = self._drain_inbox()
            return f"{inbox}\n\n{guard}"

        # Browse symbols
        if intent == "browse_symbols":
            return self._handle_symbol_browser(text, entities)

        # Add symbols
        if intent == "add_symbols" or any(k in lower for k in ["add to portfolio", "add symbols", "add pairs"]):
            return self.add_symbols_to_portfolio(text)

        # Portfolio / trading symbols
        if intent == "trading_symbols" or "portfolio" in lower:
            return self._handle_trading_symbols_intent(lower)

        return None

    def _handle_market_status(self) -> str:
        mgr = MarketSessionManager()
        now_utc = datetime.now(timezone.utc)
        portfolio_symbols = profile.symbols()
        summary = mgr.get_market_status_summary(portfolio_symbols, now_utc)
        open_syms, _ = mgr.filter_tradeable_symbols(portfolio_symbols, now_utc)
        lines = [summary]
        if open_syms:
            lines.append(
                f"\nI can scan {', '.join(open_syms)} right now. "
                f"Want me to go?"
            )
        else:
            suggestions = mgr.suggest_always_open()
            lines.append(
                f"\nYour whole portfolio is in closed markets. "
                f"Crypto trades 24/7 though — {', '.join(suggestions[:3])} are always live. "
                f"Want to add one?"
            )
        return "\n".join(lines)

    def _handle_trading_symbols_intent(self, lower: str) -> str:
        from manager.profile_manager import profile as _p
        from manager.market_sessions import MarketSessionManager
        from collections import defaultdict

        syms = _p.symbols()
        if not syms:
            return "Your portfolio is empty — say 'add EURUSD' to get started."

        mgr = MarketSessionManager()
        target_category = None
        if any(w in lower for w in ["crypto", "bitcoin", "coin", "digital"]):
            target_category = "crypto"
        elif any(w in lower for w in ["forex", "currencies", "currency", "fx"]):
            target_category = "forex"
        elif any(w in lower for w in ["metal", "metals", "gold", "silver"]):
            target_category = "metals"

        if target_category:
            filtered = [s for s in syms if target_category in mgr.get_symbol_category(s)]
            if not filtered:
                return f"No {target_category} symbols in your portfolio yet."
            emoji = {"crypto": "₿", "forex": "💱", "metals": "🥇"}.get(target_category, "📊")
            return f"{emoji} {len(filtered)} {target_category} symbol(s): {', '.join(filtered)}"

        grouped: defaultdict[str, list] = defaultdict(list)
        for s in syms:
            grouped[mgr.get_symbol_category(s)].append(s)

        CAT_EMOJI = {
            "crypto": "₿", "forex": "💱", "metals": "🥇", "commodities": "🛢",
            "stocks": "🏢", "indices_us": "🇺🇸", "indices_eu": "🇪🇺", "indices_asia": "🌏"
        }
        lines = [f"Tracking {len(syms)} symbol(s):"]
        for cat in sorted(grouped):
            items = grouped[cat]
            emoji = CAT_EMOJI.get(cat, "🔹")
            display = cat.replace("_", " ").title()
            lines.append(f"  {emoji} {display}: {', '.join(items)}")
        return "\n".join(lines)

    def _make_session_aware_greeting(self) -> str:
        from manager.market_sessions import MarketSessionManager
        session_manager = MarketSessionManager()
        watchlist = getattr(self.pm, "master_watchlist", [])
        if not watchlist:
            return "Ready — your watchlist is empty, so let's add something to track first. What do you want to trade?"
        open_syms, closed_syms = session_manager.filter_tradeable_symbols(watchlist)
        if not open_syms:
            return (
                "Markets are closed for everything in your watchlist right now. "
                "Crypto is still live 24/7 if you want to stay active."
            )
        if closed_syms:
            return (
                f"{len(open_syms)} of your symbols are open, {len(closed_syms)} are closed. "
                f"What do you want to do first?"
            )
        return "Everything in your watchlist is open. What's the plan?"

    def _is_explicit_execute(self, lower: str) -> bool:
        execute_words = [
            "execute", "place the trade", "place trade", "open trade",
            "go long", "go short", "pull the trigger", "enter now",
            "buy now", "sell now", "trade it", "take the trade",
            "get me in", "fire it", "do it", "let's go",
        ]
        return any(w in lower for w in execute_words)

    def _execute_trade_now(self, entities: dict, intent: str) -> str:
        symbol = (entities.get("symbols") or [self.memory.get("last_symbol")])[0]
        direction = entities.get("direction")

        if not symbol:
            self.pending_action = "awaiting_trade_symbol"
            self.pending_data   = {"intent": intent}
            return "Which symbol are you thinking?"

        if not direction:
            last_signal = self.memory.get("last_signal", {})
            direction   = last_signal.get("action", "BUY")
            if direction == "WAIT":
                return (
                    f"My last read on {symbol} was to wait — signal wasn't strong enough. "
                    f"Want to force a buy or sell anyway, or hold off?"
                )

        plan = self.agent.last_plan
        if plan:
            scored = next(
                (s.result for s in plan.steps if s.name == "quality_score"), {}
            ) or {}
            grade = scored.get("grade", "D")
            if grade not in ("A", "B"):
                return (
                    f"The signal on {symbol} is grade {grade} — not strong enough to pull the trigger yet. "
                    f"Waiting for a grade A or B setup."
                )

        r = profile.risk()
        allowed, reason = self.rm.is_trading_allowed(
            symbol, r.max_daily_loss, max(len(profile.symbols()), 1)
        )
        if not allowed:
            return f"Can't trade right now — {reason}"

        return self.executor.execute_trade(symbol, direction)

    def _handle_symbol_browser(self, text: str, entities: dict) -> str:
        import re
        from collections import defaultdict

        lower = text.lower()
        category = None
        if any(w in lower for w in ["forex", "currency", "currencies", "fx", "pair", "pairs"]):
            category = "forex"
        elif any(w in lower for w in ["metal", "metals", "gold", "silver"]):
            category = "metals"
        elif any(w in lower for w in ["crypto", "bitcoin", "ethereum", "btc", "eth"]):
            category = "crypto"
        elif any(w in lower for w in ["index", "indices", "stocks", "us30"]):
            category = "indices"
        elif any(w in lower for w in ["oil", "commodity", "commodities"]):
            category = "commodities"

        query = None
        search_match = re.search(
            r"(?:search|find|look\s*up|show\s*me|what\s+is\s+the\s+(?:ticker|symbol)\s+for)\s+([A-Za-z]+)",
            text, re.IGNORECASE
        )
        if search_match:
            query = search_match.group(1).upper()
        if not query and entities.get("symbols"):
            query = entities["symbols"][0][:3].upper()
        if not query:
            currencies = re.findall(
                r"\b(USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD)\b", text.upper()
            )
            if currencies:
                query = currencies[0]

        if not self.broker.connected:
            return "I need to be connected to MT5 to browse symbols."

        symbols = self.broker.search_symbols(query=query, category=category, max_results=60)

        if not symbols:
            hint = f" Nothing matched '{query}'." if query else ""
            if category:
                hint += f" No {category} symbols found."
            return (
                f"Came up empty.{hint} "
                f"Try: 'show forex pairs', 'search EURUSD', or 'what crypto is available'."
            )

        grouped: defaultdict[str, list] = defaultdict(list)
        for s in symbols:
            grouped[s["category"]].append(s)

        header = f"Found {len(symbols)} symbol(s)"
        if query:
            header += f" matching '{query}'"
        if category:
            header += f" in {category}"
        header += ":\n"

        CAT_EMOJI = {
            "forex": "💱 FOREX", "metals": "🥇 METALS", "crypto": "₿  CRYPTO",
            "indices": "📈 INDICES", "commodities": "🛢  COMMODITIES",
        }

        lines = [header]
        for cat in ["forex", "metals", "crypto", "indices", "commodities"]:
            items = grouped.get(cat, [])
            if not items:
                continue
            lines.append(f"{CAT_EMOJI.get(cat, cat.upper())} ({len(items)})")
            for item in items[:20]:
                name = item["name"]
                desc = item["description"]
                spread = item["spread_pips"]
                desc_str = f"  {desc}" if desc and desc.upper() != name.upper() else ""
                spread_str = f"  [{spread}p spread]" if spread is not None else ""
                lines.append(f"  {name}{desc_str}{spread_str}")
            if len(items) > 20:
                lines.append(f"  … and {len(items) - 20} more. Narrow it down with a keyword.")
            lines.append("")

        lines.append("To add any: 'add EURUSD to portfolio'")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-analysis action offer
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_execute(self, response: str, plan, entities: dict) -> str:
        suggested = plan.suggested_action
        if not suggested:
            return response

        parts = suggested.split()
        if len(parts) >= 3:
            direction = parts[0]
            symbol    = parts[2]
            self.memory["last_signal"] = {"action": direction, "symbol": symbol}
            self.pending_action = "confirm_trade"
            self.pending_data   = {
                "direction": direction,
                "symbol": symbol,
                "lots": float(parts[1]) if len(parts) > 1 else None
            }

            confirms = [
                f"\nWant to take the {direction} on {symbol}?",
                f"\nShall I execute that {direction} on {symbol}?",
                f"\nReady to go {direction} on {symbol}?",
            ]
            response += random.choice(confirms)

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-turn pending state machine
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_pending(self, text: str) -> tuple[bool, str]:
        lower  = text.lower().strip()
        action = self.pending_action

        YES = {"yes", "y", "go", "do it", "confirm", "execute", "yeah", "yep", "sure", "ok", "okay", "lets go", "let's go"}
        NO  = {"no", "n", "cancel", "abort", "skip", "nope", "nah", "never mind", "nevermind", "forget it"}

        if action == "confirm_trade":
            if lower in YES:
                d = self.pending_data
                resp = self.executor.execute_trade(d["symbol"], d["direction"], d.get("lots"))
                self._clear_pending()
                return True, resp
            if lower in NO:
                self._clear_pending()
                return True, random.choice([
                    "No problem, leaving it. Let me know when you want to move.",
                    "Understood, not taking it. Anything else?",
                    "Skipped. What next?",
                ])
            d = self.pending_data
            return True, (
                f"Still waiting on your call for {d['direction']} {d['symbol']}. "
                f"Yes to execute, or say cancel."
            )

        if action == "confirm_close":
            if lower in YES:
                symbol = self.pending_data.get("symbol")
                resp = self.executor.close_position(symbol)
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Leaving it open. What else?"

        if action == "confirm_close_all":
            if lower in YES:
                resp = self.executor.close_all()
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Left everything open."

        if action == "awaiting_trade_symbol":
            entities = self.nlp.extract_entities(text)
            # Also try conversational parser for symbol
            parsed = self.conv_parser.parse(text, self.working_memory)
            symbols = (parsed.symbols if parsed else []) or entities.get("symbols", [])
            if symbols:
                symbol = symbols[0]
                self.memory["last_symbol"] = symbol
                intent = self.pending_data.get("intent", "execute_trade")
                self._clear_pending()
                return True, self._execute_trade_now(entities, intent)
            if lower in NO:
                self._clear_pending()
                return True, "Got it, trade cancelled."
            return True, "Which symbol? Just name it and I'll check the signal."

        if action == "awaiting_close_symbol":
            entities = self.nlp.extract_entities(text)
            parsed = self.conv_parser.parse(text, self.working_memory)
            symbols = (parsed.symbols if parsed else []) or entities.get("symbols", [])
            if symbols:
                symbol = symbols[0]
                self._clear_pending()
                return True, self.executor.close_position(symbol)
            if lower in NO:
                self._clear_pending()
                return True, "Close cancelled."
            return True, "Which one? Just say the symbol."

        if action == "retry_micro_lot":
            if lower in YES:
                d = self.pending_data
                resp = self.executor.execute_trade(d["symbol"], d["direction"], lots=0.01)
                self._clear_pending()
                return True, resp
            self._clear_pending()
            return True, "Skipped the retry."

        return False, ""

    def _clear_pending(self):
        self.pending_action = None
        self.pending_data   = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        try:
            self.episodic_memory.flush()
            self.proactive_engine.stop()
            self.trailing_manager.stop()
            self.profit_guard.stop()
            self.position_monitor.stop()
            self.message_processor.shutdown(wait=False)
        except Exception as e:
            print(f"Error during shutdown: {e}")

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
        match  = re.search(r"change (?:my )?(.*?)\s+to\s+([0-9.]+)", text, re.IGNORECASE)
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
                if cfg_key == "cooldown_duration_minutes":
                    self.set_cooldown_duration(int(new_value))
                    self.memory["last_setting"] = target_key
                    return f"{display} set to {int(new_value)} minute(s)."
                self._update_config(cfg_key, new_value)
                self.memory["last_setting"] = target_key
                return f"{display} updated to {new_value}."
            return (
                f"I don't know that setting — '{target_key}'. "
                f"Valid ones: risk, target, stop loss, daily loss, cooldown."
            )
        return None

    def _update_config(self, key: str, value: float):
        _RISK_KEYS = {
            "risk_percentage":  "risk_pct",
            "stop_loss":        "stop_loss_pips",
            "target_profit":    "take_profit_pips",
            "max_daily_loss":   "max_daily_loss",
            "daily_goal":       "daily_goal",
            "lock_balance":     "lock_amount",
            "lock_balance_pct": "lock_pct",
        }
        if key in _RISK_KEYS:
            profile.update_risk(**{_RISK_KEYS[key]: value})
            r = profile.risk()
            self.rm.lock_guard.update(lock_amount=r.lock_amount, lock_pct=r.lock_pct_decimal)
        elif key == "cooldown_duration_minutes":
            self.broker.set_cooldown(int(value * 60))

    def set_cooldown_duration(self, minutes: int):
        seconds = max(0, int(minutes * 60))
        try:
            self.broker.set_cooldown(seconds)
            if seconds > 0:
                self._type_print(f"Cooldown set to {minutes} minute(s).")
            else:
                self._type_print("Cooldown disabled.")
        except Exception:
            self._type_print("Couldn't update the cooldown setting.")

    def add_symbols_to_portfolio(self, text: str) -> str:
        import re
        from manager.nlp_engine import SYMBOL_ALIASES

        symbols_found = re.findall(r"\b([A-Z]{3,7})\b", text.upper())
        resolved = set()
        for sym in symbols_found:
            if sym in SYMBOL_ALIASES:
                resolved.add(SYMBOL_ALIASES[sym])
            elif len(sym) >= 6:
                resolved.add(sym)
            elif len(sym) == 3:
                if sym == "USD":
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
            return "Didn't recognise any symbols there. Try: 'add EURUSD' or 'add bitcoin'."

        existing = set(profile.symbols())
        added = set()
        for sym in resolved:
            if profile.add_symbol(sym):
                added.add(sym)
        if not added:
            return f"All of those are already in your portfolio: {', '.join(sorted(existing))}"

        updated = profile.symbols()
        return (
            f"Added {', '.join(sorted(added))}. "
            f"Portfolio now tracking {len(updated)} symbol(s): {', '.join(updated)}."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Notification inbox
    # ─────────────────────────────────────────────────────────────────────────

    def _drain_inbox(self) -> str:
        with self.inbox_lock:
            if not self.notification_inbox:
                return "Nothing new — inbox is clean."
            items = list(self.notification_inbox)
            self.notification_inbox.clear()

        lines = [f"{len(items)} unread event(s):"]
        for item in items:
            icon = {"trade_executed": "🟢", "critical": "⚠️"}.get(item["priority"], "ℹ️")
            lines.append(f"  {icon} {item['msg']}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Agent step callback
    # ─────────────────────────────────────────────────────────────────────────

    def _step_callback(self, name: str, description: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [ARIA ▸]: {description}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Startup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _broker_login(self):
        if self.broker.connected:
            return

        data = self.profile_manager.load_credentials()
        login, password, server = (
            data.get("login"), data.get("password"), data.get("server")
        )

        if login and password and server:
            if self.broker.connect(login=login, password=password, server=server):
                return
            self._type_print("Saved credentials didn't work — let's try again.")

        while not self.broker.connected:
            try:
                login = int(input("[ARIA]: MT5 Account Number: ").strip())
            except ValueError:
                self._type_print("That needs to be a number.")
                continue
            if len(str(login)) < 5:
                self._type_print("That account number looks too short.")
                continue

            password = getpass.getpass("[ARIA]: Password: ")
            if not password:
                self._type_print("Password can't be empty.")
                continue

            server = input("[ARIA]: Broker Server (e.g. MetaQuotes-Demo): ").strip()
            if not server:
                self._type_print("Server name is required.")
                continue

            if self.broker.connect(login=login, password=password, server=server):
                self.profile_manager.save_credentials(login, password, server)
                self._type_print(f"Connected — account #{self.broker.login} is live.")
            else:
                self._type_print("Couldn't connect. Double-check the credentials and make sure MT5 is running.")

    def _load_or_setup_config(self):
        if self.PROFILE_FILE.exists():
            try:
                raw = json.loads(self.PROFILE_FILE.read_text())
                if "portfolio" in raw and raw["portfolio"].get("symbols"):
                    return
            except Exception:
                pass

    def _setup_config(self):
        print("\n" + "=" * 45 + "\n🤖  ARIA FIRST-TIME SETUP\n" + "=" * 45)
        user_input = input(
            "[ARIA]: What's your goal? "
            "(e.g. 'Risk 1%, make $20 daily on EURUSD, GBPUSD'): "
        ).strip()

        text_upper = user_input.upper()
        symbols = list(set(re.findall(r"\b[A-Z]{6}\b", text_upper)))
        risk_m  = re.search(r"(\d+(?:\.\d+)?)\s*%", user_input)
        profit_m = re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:daily|per day)?", user_input)

        symbols    = symbols or ["EURUSD"]
        daily_goal = float(profit_m.group(1)) if profit_m else 10.0
        target_profit = round(daily_goal / max(len(symbols), 1), 2)
        risk_pct = float(risk_m.group(1)) if risk_m else 1.0

        cfg = {
            "_version": 2,
            "portfolio": {
                "symbols": symbols,
                "preferred_timeframes": ["H1"],
                "asset_classes": {},
                "strategy_mapping": {
                    "default": "Mean_Reversion",
                    "asset_class_defaults": {
                        "forex": "Mean_Reversion", "metals": "Momentum",
                        "crypto": "Momentum", "indices": "Breakout",
                    },
                    "symbol_overrides": {}
                }
            },
            "risk": {
                "defaults": {
                    "risk_pct": risk_pct, "stop_loss_pips": 20.0,
                    "take_profit_pips": target_profit, "max_daily_loss": daily_goal * 2,
                    "daily_goal": daily_goal, "cooldown_minutes": 5,
                    "lock_amount": 0.0, "lock_pct": 0.0,
                },
                "symbol_overrides": {},
                "drawdown_recovery": {
                    "moderate_threshold": 0.5, "moderate_scale": 0.5,
                    "critical_threshold": 0.8, "critical_scale": 0.25
                }
            },
            "broker": {
                "max_open_trades": 3, "min_margin_level": 150.0,
                "spread_tolerance_pips": 3.0, "magic_number": 1000, "slippage_points": 20
            },
            "sessions": {
                "avoid_asian_session": True, "avoid_friday_close": True,
                "asian_session_pairs": ["USDJPY", "AUDUSD", "NZDUSD", "CADJPY", "CHFJPY", "GBPJPY"]
            },
            "scanner": {
                "interval_seconds": 3, "dry_run": False,
                "mtf_min_alignment": 0.50, "volatility_spike_atr": 2.5, "dead_volume_ratio": 0.3
            }
        }
        self.PROFILE_FILE.parent.mkdir(exist_ok=True)
        self.PROFILE_FILE.write_text(json.dumps(cfg, indent=4))
        self._type_print(
            f"Config saved — tracking {len(symbols)} symbol(s), "
            f"{risk_pct}% risk, ${target_profit} target per symbol."
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
                if lower in ("quit", "exit", "bye", "goodbye", "later", "done"):
                    self._type_print(random.choice([
                        "Later. Good luck out there.",
                        "Logging off. Trade safe.",
                        "Done for now. Catch you next session.",
                    ]))
                    break

                try:
                    response = self.process_message(inp)
                except Exception as exc:
                    response = f"Something went wrong on my end: {exc}. Try again."

                if response:
                    self._type_print(response)

        except KeyboardInterrupt:
            self._type_print("Shutting down.")

    # ─────────────────────────────────────────────────────────────────────────
    # Output helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _type_print(self, msg: str, delay: float = 0.010) -> None:
        ts    = datetime.now().strftime("%H:%M:%S")
        clean = msg.replace("[Bot]: ", "").replace("[ARIA]: ", "").strip()
        prefix = f"[{ts}] [ARIA]: "

        if sys.stdout.isatty():
            sys.stdout.write(prefix)
            for char in clean:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay)
            sys.stdout.write("\n")
        else:
            print(f"{prefix}{clean}")

    def bot_print(self, msg: str):
        self._type_print(msg)

    def chat(self, user_message: str, action_result: str = None) -> str:
        if action_result:
            return f"Done — {action_result}"
        return self.process_message(user_message)