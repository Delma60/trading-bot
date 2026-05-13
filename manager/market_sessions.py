"""
manager/market_sessions.py — Market Session Awareness

Knows which markets are open right now and what's tradeable.
"""

from datetime import datetime, timezone
from typing import Optional
from manager.symbol_registry import SymbolRegistry


class MarketSession:
    def __init__(
        self,
        name:             str,
        open_hour_utc:    int,
        close_hour_utc:   int,
        weekdays_only:    bool = True,
        always_open:      bool = False,
        all_day_weekdays: bool = False,
    ):
        self.name             = name
        self.open_hour        = open_hour_utc
        self.close_hour       = close_hour_utc
        self.weekdays_only    = weekdays_only
        self.always_open      = always_open
        self.all_day_weekdays = all_day_weekdays

    def is_open(self, now_utc: datetime = None) -> bool:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        if self.always_open:
            return True
        weekday = now_utc.weekday()
        hour    = now_utc.hour

        if self.all_day_weekdays:
            if weekday == 5:                        # Saturday always closed
                return False
            if weekday == 6:                        # Sunday open after open_utc
                return hour >= self.open_hour if self.open_hour else hour >= 22
            if weekday == 4:                        # Friday close at close_utc
                return hour < self.close_hour if self.close_hour else hour < 22
            return True                             # Mon–Thu always open

        if self.weekdays_only and weekday >= 5:
            return False
        if self.open_hour < self.close_hour:
            return self.open_hour <= hour < self.close_hour
        else:                                       # wraps midnight
            return hour >= self.open_hour or hour < self.close_hour


# ── Friendly display names (module-level, used by MarketSessionManager) ──────
_FRIENDLY_NAMES = {
    "crypto":       "Crypto (BTC, ETH…)",
    "forex":        "Forex (EUR/USD, GBP/USD…)",
    "metals":       "Metals (Gold, Silver…)",
    "indices_us":   "US Indices (US30, NAS100…)",
    "indices_eu":   "EU Indices (GER40, UK100…)",
    "indices_asia": "Asia Indices (JPN225, AUS200…)",
    "commodities":  "Commodities (Oil, Gas…)",
    "stocks":       "US Stocks (AAPL, TSLA…)",
}

# Hardcoded fallback if profile has no market_sessions section
_FALLBACK_SESSIONS: dict[str, MarketSession] = {
    "forex":        MarketSession("Forex",        0,  0,  weekdays_only=False, all_day_weekdays=True),
    "metals":       MarketSession("Metals",       23, 22, weekdays_only=False, all_day_weekdays=True),
    "crypto":       MarketSession("Crypto",       0,  0,  weekdays_only=False, always_open=True),
    "indices_us":   MarketSession("US Markets",   13, 21, weekdays_only=True),
    "indices_eu":   MarketSession("EU Markets",   7,  17, weekdays_only=True),
    "indices_asia": MarketSession("Asia Markets", 0,  9,  weekdays_only=True),
    "commodities":  MarketSession("Commodities",  1,  21, weekdays_only=True),
    "stocks":       MarketSession("US Stocks",    13, 21, weekdays_only=True),
}


class MarketSessionManager:
    """
    Single source of truth for session awareness across the entire bot.

    Usage
    -----
        mgr = MarketSessionManager()

        # Is this symbol tradeable right now?
        ok, reason = mgr.is_symbol_tradeable("EURUSD")

        # Split a watchlist into open / closed buckets
        open_syms, closed_syms = mgr.filter_tradeable_symbols(["EURUSD", "BTCUSD"])

        # Market status summary
        summary = mgr.get_market_status_summary(["EURUSD", "BTCUSD", "ETHUSD"])
    """

    CATEGORY_MAP: dict[str, list[str]] = {
        "crypto":       ["BTC", "ETH", "LTC", "XBT", "DOGE", "ADA", "SOL",
                         "XRP", "BNB", "MATIC", "DOT", "LINK", "UNI", "AVAX",
                         "SHIB", "PEPE", "NEAR", "FET", "INJ"],
        "metals":       ["XAU", "XAG", "XPT", "XPD"],
        "indices_us":   ["US30", "US500", "NAS100", "SPX", "NDX", "DOW",
                         "SP500", "US2000", "VIX"],
        "indices_eu":   ["GER40", "UK100", "FRA40", "ESP35", "DAX",
                         "FTSE", "CAC", "EUSTX50"],
        "indices_asia": ["JPN225", "AUS200", "HKG33", "NKY", "CN50", "SG30"],
        "commodities":  ["USOIL", "BRENT", "NGAS", "CORN", "WHEAT",
                         "COFFEE", "COCOA", "SUGAR"],
        "stocks":       ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "META",
                         "NVDA", "NFLX", "BABA"],
    }

    def __init__(self):
        self.SESSIONS       = self._build_sessions()
        self.FRIENDLY_NAMES = _FRIENDLY_NAMES

    def _build_sessions(self) -> dict[str, MarketSession]:
        """
        Load session hours from profile.json.
        Falls back to hardcoded defaults if the section is missing.
        """
        try:
            from manager.profile_manager import profile
            ms_config = profile.market_sessions()
            if not ms_config.sessions:
                return _FALLBACK_SESSIONS

            name_map = {
                "forex":        "Forex",
                "metals":       "Metals",
                "crypto":       "Crypto",
                "indices_us":   "US Markets",
                "indices_eu":   "EU Markets",
                "indices_asia": "Asia Markets",
                "commodities":  "Commodities",
                "stocks":       "US Stocks",
            }
            built: dict[str, MarketSession] = {}
            for category, cfg in ms_config.sessions.items():
                built[category] = MarketSession(
                    name             = name_map.get(category, category.title()),
                    open_hour_utc    = cfg.open_utc,
                    close_hour_utc   = cfg.close_utc,
                    weekdays_only    = cfg.weekdays_only,
                    always_open      = cfg.always_open,
                    all_day_weekdays = cfg.all_day_weekdays,
                )

            # Fill any missing categories with fallback
            for cat, sess in _FALLBACK_SESSIONS.items():
                if cat not in built:
                    built[cat] = sess

            return built

        except Exception:
            return _FALLBACK_SESSIONS

    def get_symbol_category(self, symbol: str) -> str:
        """Return the market category for a broker symbol."""
        sym = symbol.upper()

        for cat, keywords in self.CATEGORY_MAP.items():
            if any(sym.startswith(kw) or kw in sym for kw in keywords):
                return cat

        forex_currencies = ["USD", "EUR", "GBP", "JPY", "CHF",
                            "CAD", "AUD", "NZD", "SEK", "NOK", "DKK"]
        if any(sym[:3] in forex_currencies or sym[3:6] in forex_currencies
               for _ in [None]):
            return "forex"

        return "forex"

    def is_symbol_tradeable(
        self, symbol: str, now_utc: datetime = None
    ) -> tuple[bool, str]:
        """Return (tradeable, human-readable reason)."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        category = self.get_symbol_category(symbol)
        session  = self.SESSIONS.get(category, self.SESSIONS["forex"])

        if session.is_open(now_utc):
            return True, f"{session.name} is open"

        weekday = now_utc.weekday()
        if weekday >= 5:
            day_name = "Saturday" if weekday == 5 else "Sunday"
            return False, f"{symbol} market closed — {day_name}. Try crypto (24/7)."

        return False, (
            f"{symbol} is outside {session.name} trading hours "
            f"({session.open_hour:02d}:00–{session.close_hour:02d}:00 UTC)."
        )

    def filter_tradeable_symbols(
        self, symbols: list[str], now_utc: datetime = None
    ) -> tuple[list[str], list[str]]:
        """Split a watchlist into (open_symbols, closed_symbols)."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        open_syms, closed_syms = [], []
        for sym in symbols:
            ok, _ = self.is_symbol_tradeable(sym, now_utc)
            (open_syms if ok else closed_syms).append(sym)
        return open_syms, closed_syms

    def get_open_categories(self, now_utc: datetime = None) -> list[str]:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        return [cat for cat, sess in self.SESSIONS.items() if sess.is_open(now_utc)]

    def get_closed_categories(self, now_utc: datetime = None) -> list[str]:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        return [cat for cat, sess in self.SESSIONS.items() if not sess.is_open(now_utc)]

    def get_market_status_summary(
        self, portfolio_symbols: list[str] = None, now_utc: datetime = None
    ) -> str:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        weekday  = now_utc.weekday()
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"][weekday]
        time_str = now_utc.strftime("%H:%M UTC")

        lines = [f"Market status — {day_name} {time_str}:"]

        for cat in ["crypto", "forex", "metals", "indices_us",
                    "indices_eu", "indices_asia", "commodities", "stocks"]:
            sess = self.SESSIONS[cat]
            icon = "✅" if sess.is_open(now_utc) else "❌"
            lines.append(f"  {icon} {self.FRIENDLY_NAMES[cat]}")

        if portfolio_symbols:
            open_syms, closed_syms = self.filter_tradeable_symbols(
                portfolio_symbols, now_utc
            )
            if open_syms:
                lines.append(f"\nTradeable now: {', '.join(open_syms)}")
            if closed_syms:
                lines.append(f"Closed: {', '.join(closed_syms)}")
            if not open_syms:
                lines.append("\nNothing in your portfolio is tradeable right now.")
                lines.append("💡 Add crypto pairs (BTCUSD, ETHUSD) to trade 24/7.")

        return "\n".join(lines)

    def suggest_always_open(self) -> list[str]:
        """Return a short list of commonly available crypto pairs."""
        return SymbolRegistry().get_always_open()

    def get_next_open_time(self, symbol: str, now_utc: datetime = None) -> str:
        """Return a human-friendly string for when a closed market reopens."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        category = self.get_symbol_category(symbol)
        ok, _    = self.is_symbol_tradeable(symbol, now_utc)
        if ok:
            return "now (market is open)"

        weekday = now_utc.weekday()

        if category == "crypto":
            return "now (crypto never closes)"

        if category in ("forex", "metals"):
            if weekday == 5:
                return "Sunday ~22:00 UTC (forex Sunday open)"
            if weekday == 6:
                hours_left = max(0, 22 - now_utc.hour)
                return f"today at 22:00 UTC (~{hours_left}h from now)"
            return "now (forex is open on weekdays)"

        if category in ("indices_us", "stocks"):
            if weekday >= 5:
                return "Monday 13:30 UTC (NYSE open)"
            sess = self.SESSIONS[category]
            if now_utc.hour < sess.open_hour:
                return f"today at {sess.open_hour}:30 UTC"
            return "tomorrow 13:30 UTC"

        return "Monday (next weekday)"