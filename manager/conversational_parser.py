"""
manager/conversational_parser.py — Natural Language Understanding Layer

Sits BEFORE the NLP intent classifier. Understands natural, casual speech
and resolves it to intent + context without requiring command-like phrasing.

The goal: "what do you reckon about gold?" should work just as well as "analyze XAUUSD"
"""

import re
from typing import Optional
from dataclasses import dataclass, field
import random

@dataclass
class ParsedIntent:
    """Result of conversational parsing."""
    intent: str
    confidence: float
    symbols: list = field(default_factory=list)
    direction: Optional[str] = None
    is_question: bool = False
    is_casual: bool = False
    raw_input: str = ""
    context_used: bool = False  # True if we filled gaps from conversation memory


# ── Natural language pattern groups ────────────────────────────────────────────
# Each group maps natural speech patterns to an intent.
# Ordered by specificity (most specific first).

_PATTERN_GROUPS = [

    # ── Analysis / opinion requests ──────────────────────────────────────────
    {
        "intent": "analyze_symbol",
        "patterns": [
            r"\bwhat(?:'s| is| do you) (?:think|reckon|say|suggest) about\b",
            r"\bhow(?:'s| is| does) (?:.{1,30}?) look(?:ing)?(?:\?|$)",
            r"\bis (?:.{1,30}?) (?:worth|good|looking|a buy|a sell)\b",
            r"\b(?:any|got any) (?:thoughts?|take|view|read|edge) on\b",
            r"^(?:thoughts?|take|view) on\b",
            r"\bwhat(?:'s| is) (?:the|your) (?:read|take|view|call|bias|edge) on\b",
            r"\b(?:check|look at|look into|have a look at|inspect)\b",
            r"\b(?:analyze|analyse|analysis|break down)\b",
            r"\bwhat(?:'s| is) (?:it|this|that pair) doing\b",
            r"^(?:how|what) about (?:.{1,30}?)(?:\?|$)",
            r"\bgive me(?: a| the| your)? (?:read|take|signal|view|analysis)\b",
            r"^(?:tell|show) me about\b",
            r"\b(?:chart|chart out|pull up|open chart)\b",
            r"\bworth (?:a look|trading|watching|a punt)(?:\?|$)",
        ],
    },

    # ── Trade execution ───────────────────────────────────────────────────────
    {
        "intent": "execute_trade",
        "patterns": [
            r"\b(?:i want to|let'?s|let me|ready to) (?:buy|sell|trade|take|enter|get in)\b",
            r"\b(?:pull|fire|squeeze) the trigger\b",
            r"\bget me in\b",
            r"\btake (?:it|the trade|the position|the setup)\b",
            r"\b(?:open|start|place) a (?:position|trade|order)\b",
            r"^(?:yes|yep|yeah|do it)[,.]?\s*(?:take it|do it|go|execute|trade|enter)\b",
        ],
    },

    # ── Buy intent ────────────────────────────────────────────────────────────
    {
        "intent": "open_buy",
        "patterns": [
            r"\b(?:buy|long|go long|get long|slap the ask)\b",
            r"\bi(?:'m| am) (?:going|looking to|thinking about) (?:long|buying)\b",
            r"\b(?:bullish|bull|long bias) on\b",
        ],
    },

    # ── Sell intent ───────────────────────────────────────────────────────────
    {
        "intent": "open_sell",
        "patterns": [
            r"\b(?:sell|short|go short|get short|hit the bid)\b",
            r"\bi(?:'m| am) (?:going|looking to|thinking about) (?:short|selling)\b",
            r"\b(?:bearish|bear|short bias) on\b",
        ],
    },

    # ── Position management ───────────────────────────────────────────────────
    {
        "intent": "active_positions",
        "patterns": [
            r"\b(?:how(?:'s| are)? my|what(?:'s| are)? my|show(?: me)? my) (?:trades?|positions?|exposure|book|bags?)\b",
            r"\bwhat(?:'s)? (?:open|running|live|active)\b",
            r"\bany(?:thing)? (?:open|running|live|on)\b",
            r"\bwhat am i (?:in|holding|carrying)\b",
            r"^(?:check|review)(?: my)? (?:positions?|trades?|exposure|book)\b",
        ],
    },

    # ── Close position ────────────────────────────────────────────────────────
    {
        "intent": "close_position",
        "patterns": [
            r"\b(?:get me out|i want out|close)(?: it| (?:the|my) (?:trade|position|pair))\b",
            r"\b(?:exit|flat(?:ten)?)(?: the| my)? (?:trade|position|this|pair)\b",
            r"\b(?:cut|kill|dump)(?: the| my| it)? (?:trade|position|this|bag)\b",
            r"\btake (?:the )?(?:loss|profit|l|w)\b",
            r"\bbook (?:it|the profit|the gains)\b",
            r"^(?:enough|done with this|get out)\b",
        ],
    },

    # ── Close all ─────────────────────────────────────────────────────────────
    {
        "intent": "close_all",
        "patterns": [
            r"\b(?:close|exit|flat(?:ten)?) (?:everything|all|the lot|all of it|all positions)\b",
            r"\b(?:nuke|wipe|clear)(?: the| my)? (?:book|positions?|everything)\b",
            r"\b(?:get me|go) flat\b",
        ],
    },

    # ── Account / balance ─────────────────────────────────────────────────────
    {
        "intent": "account_summary",
        "patterns": [
            r"\b(?:how much|what(?:'s| is)) (?:in my|my) (?:account|balance|equity|pot|margin)\b",
            r"^(?:check|show)(?: me)? (?:my )?(?:balance|equity|account|funds?|free margin)\b",
            r"\bwhere(?:'s)? my (?:money|account|balance)\b",
            r"\bam i (?:up|down)(?: overall| today)?\b",
            r"\bwhat(?:'s| is) (?:my|the) (?:balance|equity|margin|pnl)\b",
        ],
    },

    # ── Performance / history ─────────────────────────────────────────────────
    {
        "intent": "trade_history",
        "patterns": [
            r"\bhow(?:'s| did| have) (?:i|we|it) (?:done?|perform(?:ed)?|gone?)\b",
            r"\bwhat(?:'s|'ve| is| have) i (?:made|earned|lost|done|got)\b",
            r"^(?:today(?:'s)?|daily) (?:pnl|results?|performance|trades?|log)\b",
            r"^(?:my )?(?:pnl|profit|loss|results?|performance|history|stats)\b",
            r"\bhow(?:'d| did| was) (?:today|the day|the session)\b",
        ],
    },

    # ── Risk / drawdown ───────────────────────────────────────────────────────
    {
        "intent": "risk_management",
        "patterns": [
            r"\bhow(?:'s| is) (?:my|the) (?:risk|exposure|drawdown|dd)\b",
            r"\bam i (?:over(?:exposed|leveraged)|at risk|in trouble)\b",
            r"\bwhat(?:'s| is) (?:my|the) (?:risk|exposure|drawdown|dd|margin level)\b",
        ],
    },

    # ── Market scan ───────────────────────────────────────────────────────────
    {
        "intent": "bulk_scan",
        "patterns": [
            r"\bany(?:thing)? (?:good|worth|interesting|looking|out there|to trade)\b",
            r"\b(?:find|hunt for|look for) (?:a )?(?:trade|setup|opportunity|entry)\b",
            r"^(?:scan|sweep|check)(?: the| my| for)? (?:market|portfolio|everything|pairs?|symbols?|setups?)?\b",
            r"\bwhat(?:'s| should i)? (?:look at|trade|be trading|be watching)\b",
            r"^(?:show me )?(?:the )?(?:opportunities|setups|entries|signals)\b",
            r"^\b(?:go|start|run|run scan)\b$",
        ],
    },

    # ── Market status ─────────────────────────────────────────────────────────
    {
        "intent": "market_status",
        "patterns": [
            r"\bwhat(?:'s| is) (?:open|tradeable|available|on|live)\b",
            r"\bcan i trade (?:now|today|yet)\b",
            r"^\b(?:are|is) (?:the )?markets? (?:open|live|active)\b",
        ],
    },

    # ── News ──────────────────────────────────────────────────────────────────
    {
        "intent": "news_update",
        "patterns": [
            r"\bany(?:thing)? (?:in the|on the)? news\b",
            r"\bwhat(?:'s| is) (?:in the|happening in the|going on in the) news\b",
            r"\b(?:economic|market|macro) (?:events?|calendar|news|data)\b",
            r"\bany (?:events?|data|releases?) (?:today|this week|coming up)\b",
            
        ],
    },

    # ── Price check ───────────────────────────────────────────────────────────
    {
        "intent": "get_price",
        "patterns": [
            r"\bwhat(?:'s| is) (?:.{1,20}?) (?:at|trading at|priced at|sitting at|going for)\b",
            r"^(?:current |live )?(?:price|rate|quote|level) (?:of|on|for) \b",
            r"\bwhere(?:'s| is) (?:.{1,20}?) (?:at|trading|sitting)(?:\?|$)",
            r"^(?:how much is|quote) (?:.{1,20}?)(?:\?|$)",
        ],
    },

    # ── Greetings & small talk ────────────────────────────────────────────────
    {
        "intent": "greeting",
        "patterns": [
            r"^(?:hey|hi|hello|morning|evening|afternoon|sup|yo|what'?s up|good (?:morning|afternoon|evening))\b",
            r"^(?:how(?:'s| are) (?:it|things|you|we|the market))\b",
            r"^(?:ready|let'?s (?:go|start|do this|get going|get to work))\b",
            r"^(?:what(?:'s| is) (?:the plan|going on|happening))\b",
        ],
    },

    # ── Acknowledgement / reaction ────────────────────────────────────────────
    {
        "intent": "acknowledgment",
        "patterns": [
            r"^(?:ok(?:ay)?|got it|understood|noted|makes sense|fair enough|alright|right|cool)\b",
            r"^(?:thanks?|cheers|nice one|good call|perfect|solid)\b",
        ],
    },

    # ── Gratitude ─────────────────────────────────────────────────────────────
    {
        "intent": "gratitude",
        "patterns": [
            r"^(?:thank(?:s| you)|appreciate it|good (?:job|work|one)|well done)\b",
        ],
    },

    # ── Symbols browse ────────────────────────────────────────────────────────
    {
        "intent": "browse_symbols",
        "patterns": [
            r"\bwhat (?:pairs?|symbols?|instruments?|(?:else )?can i trade|do you have|forex|crypto|metals)\b",
            r"^(?:show|list)(?: me)? (?:all |available )?(?:pairs?|symbols?|instruments?|assets?)\b",
            r"\bwhat'?s (?:available|on offer|on the broker)\b",
            r"^(?:search|find|look up)(?: a)? symbol\b",
        ],
    },
]

# ── Ambiguous filler patterns that should trigger clarification ───────────────
_AMBIGUOUS_PATTERNS = [
    r"^(?:yes|no|maybe|sure|okay|yeah|nah|nope|hmm|hm|erm|uhh?)\s*$",
    r"^(?:it|that|this|them|those)\s*$",
]

# ── Close-to-close language (when direction is clear from phrasing) ────────────
_CLOSE_NOW_PHRASES = [
    "close it", "get out", "exit", "cut it", "kill it", "take profits",
    "book it", "i'm out", "flat", "done", "enough", "close now",
]

# ── Symbol aliases (shorthand → canonical) ────────────────────────────────────
_QUICK_ALIASES = {
    "gold": "XAUUSD", "silver": "XAGUSD", "euro": "EURUSD", "cable": "GBPUSD",
    "pound": "GBPUSD", "yen": "USDJPY", "swissy": "USDCHF", "loonie": "USDCAD",
    "aussie": "AUDUSD", "kiwi": "NZDUSD", "bitcoin": "BTCUSD", "btc": "BTCUSD",
    "eth": "ETHUSD", "ethereum": "ETHUSD", "oil": "USOIL", "dow": "US30",
    "nasdaq": "US100", "sp500": "US500", "dax": "GER40", "ftse": "UK100",
    "natgas": "NGAS", "xau": "XAUUSD", "xag": "XAGUSD",
}


class ConversationalParser:
    """
    Pre-classifies natural speech into intents before the NLP engine runs.
    Understands casual, ambiguous, and context-dependent language.
    """

    def __init__(self):
        # Pre-compile all patterns for speed
        self._compiled = []
        for group in _PATTERN_GROUPS:
            compiled_patterns = [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in group["patterns"]
            ]
            self._compiled.append({
                "intent": group["intent"],
                "patterns": compiled_patterns,
            })

        self._ambiguous = [
            re.compile(p, re.IGNORECASE) for p in _AMBIGUOUS_PATTERNS
        ]

    def parse(self, text: str, working_memory=None) -> Optional[ParsedIntent]:
        """
        Attempt to parse natural language into an intent.
        Returns None if the text should fall through to the NLP engine.

        working_memory is used to fill context gaps (e.g. "what about it?" → last symbol)
        """
        text_clean = text.strip()
        lower = text_clean.lower()

        # 1. Detect symbols in the message
        symbols = self._extract_symbols(lower, text_clean)

        # 2. Detect direction (buy/sell bias)
        direction = self._extract_direction(lower)

        # 3. Check if it's ambiguous/incomplete and context is needed
        if self._is_ambiguous(lower) and working_memory:
            return self._resolve_with_context(text_clean, lower, working_memory, symbols, direction)

        # 4. Match against natural language patterns
        for group in self._compiled:
            for pattern in group["patterns"]:
                if pattern.search(lower):
                    intent = group["intent"]

                    # Refine execute intent based on direction
                    if intent == "execute_trade" and direction == "BUY":
                        intent = "open_buy"
                    elif intent == "execute_trade" and direction == "SELL":
                        intent = "open_sell"

                    # If no symbol found and working memory has one, use it
                    context_used = False
                    if not symbols and working_memory and working_memory.last_symbol:
                        if intent in ("analyze_symbol", "get_price", "open_buy",
                                      "open_sell", "execute_trade", "close_position"):
                            symbols = [working_memory.last_symbol]
                            context_used = True

                    return ParsedIntent(
                        intent=intent,
                        confidence=0.85,
                        symbols=symbols,
                        direction=direction,
                        is_question=text_clean.endswith("?"),
                        is_casual=True,
                        raw_input=text_clean,
                        context_used=context_used,
                    )

        # 5. Symbol-only message (e.g. just "EURUSD" or "gold?")
        if symbols and len(text_clean.split()) <= 3:
            intent = "open_buy" if direction == "BUY" else \
                     "open_sell" if direction == "SELL" else "analyze_symbol"
            return ParsedIntent(
                intent=intent,
                confidence=0.75,
                symbols=symbols,
                direction=direction,
                is_question=text_clean.endswith("?"),
                is_casual=True,
                raw_input=text_clean,
            )

        # 6. Very short affirmations in context of pending action
        if lower in ("yes", "y", "yep", "yeah", "go", "do it", "confirmed", "confirm"):
            return ParsedIntent(
                intent="__confirm__",
                confidence=1.0,
                raw_input=text_clean,
                is_casual=True,
            )

        if lower in ("no", "n", "nope", "nah", "cancel", "never mind", "nevermind", "forget it", "skip"):
            return ParsedIntent(
                intent="__cancel__",
                confidence=1.0,
                raw_input=text_clean,
                is_casual=True,
            )

        # 7. Close-now language
        if any(phrase in lower for phrase in _CLOSE_NOW_PHRASES):
            if symbols or (working_memory and working_memory.last_symbol):
                return ParsedIntent(
                    intent="close_position",
                    confidence=0.85,
                    symbols=symbols or ([working_memory.last_symbol] if working_memory else []),
                    raw_input=text_clean,
                    is_casual=True,
                )

        # Couldn't parse — return None to fall through to NLP engine
        return None

    def _extract_symbols(self, lower: str, original: str) -> list:
        """Extract trading symbols from natural language."""
        found = []

        # Check quick aliases first
        for alias, canonical in _QUICK_ALIASES.items():
            if re.search(rf'\b{re.escape(alias)}\b', lower):
                if canonical not in found:
                    found.append(canonical)

        # Then look for 6-7 char uppercase tickers
        for match in re.findall(r'\b([A-Z]{6,7})\b', original):
            if match not in found:
                found.append(match)

        return found

    def _extract_direction(self, lower: str) -> Optional[str]:
        """Detect buy/sell bias from casual language."""
        buy_words = [r'\bbuy\b', r'\blong\b', r'\bbullish\b', r'\bgo long\b', r'\bget long\b']
        sell_words = [r'\bsell\b', r'\bshort\b', r'\bbearish\b', r'\bgo short\b', r'\bget short\b']

        for pattern in buy_words:
            if re.search(pattern, lower):
                return "BUY"
        for pattern in sell_words:
            if re.search(pattern, lower):
                return "SELL"
        return None

    def _is_ambiguous(self, lower: str) -> bool:
        """Check if the message is too vague to classify without context."""
        for pattern in self._ambiguous:
            if pattern.match(lower):
                return True
        # Very short messages that aren't clear commands
        if len(lower.split()) <= 2 and not any(
            kw in lower for kw in ["go", "scan", "run", "yes", "no", "ok", "done"]
        ):
            return True
        return False

    def _resolve_with_context(
        self, text: str, lower: str, wm, symbols: list, direction: Optional[str]
    ) -> Optional[ParsedIntent]:
        """Use working memory to resolve an ambiguous message."""
        last_intent = getattr(wm, "last_intent", None)
        last_symbol = getattr(wm, "last_symbol", None)

        # "what about it?" or "and gold?" after analysis → analyze the new or same symbol
        if symbols:
            return ParsedIntent(
                intent="analyze_symbol",
                confidence=0.80,
                symbols=symbols,
                direction=direction,
                is_casual=True,
                raw_input=text,
                context_used=True,
            )

        # Pure follow-up with no new info → repeat last action on same symbol
        if last_symbol and last_intent:
            return ParsedIntent(
                intent=last_intent,
                confidence=0.65,
                symbols=[last_symbol],
                direction=direction,
                is_casual=True,
                raw_input=text,
                context_used=True,
            )

        return None

    def get_natural_fallback(self, text: str, working_memory=None) -> str:
        """
        Generate a natural, conversational response when we can't understand the message.
        Much warmer than "I didn't catch that."
        """
        lower = text.lower()

        # Emotional / situational responses
        if any(w in lower for w in ["stressed", "worried", "nervous", "scared", "anxious"]):
            return "I hear you — markets can be stressful. Want me to check your current exposure so we can see what's actually at risk?"

        if any(w in lower for w in ["bored", "nothing to do", "quiet", "slow"]):
            return "Quiet market day? I can scan the portfolio and see if anything's setting up. Want me to take a look?"

        if any(w in lower for w in ["bad day", "rough", "losing", "down"]):
            return "Rough session? Let me pull your P&L so we can see exactly where things stand — sometimes the picture is better than it feels."

        if any(w in lower for w in ["good day", "great", "killing it", "up", "winning"]):
            return "Nice — let me pull your numbers so we can see exactly how well you're doing. Worth locking any of it in?"

        if "help" in lower:
            return "Sure. I can analyze any symbol, scan for setups, check your account, manage positions, or watch the news. What do you need?"

        # Symbol mentioned but no clear intent
        from manager.nlp_engine import SYMBOL_ALIASES
        for alias in SYMBOL_ALIASES:
            if alias.lower() in lower:
                canonical = SYMBOL_ALIASES[alias]
                return f"You mentioned {alias} — want me to pull the signal on {canonical}, or were you thinking of taking a position?"

        # Context-aware follow-up
        if working_memory and working_memory.last_symbol:
            sym = working_memory.last_symbol
            return f"Not quite sure what you mean — were you asking about {sym}, or something else entirely?"

        # Generic
        options = [
            "Not sure I caught that. You can ask me to analyze a symbol, check your account, scan for trades, or manage positions.",
            "Hmm, I didn't quite follow. What are you looking to do — analyze something, check positions, or scan the market?",
            "Could you say that differently? I'm best when you ask me to analyze a pair, check your balance, or find a trade.",
        ]
        
        return random.choice(options)