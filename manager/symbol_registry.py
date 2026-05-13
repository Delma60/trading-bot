"""
manager/symbol_registry.py — Broker-Aware Symbol Registry

Single source of truth for every symbol used anywhere in ARIA.
All other modules that previously kept their own hardcoded symbol lists
(ArbitrageStrategy, main.py fallbacks, etc.) import from here instead.

Architecture
------------
SymbolRegistry wraps Trader.search_symbols() with:
  - One-time (lazy) load on first access
  - Per-category TTL-cached refresh (default 5 min)
  - Synchronous fallback list for when the broker is disconnected
  - Thread-safe reads

Usage
-----
    from manager.symbol_registry import SymbolRegistry

    registry = SymbolRegistry(broker)

    # All tradeable forex symbols on the connected broker
    forex = registry.get_universe("forex")

    # Best candidates for arbitrage (correlated, liquid)
    arb_universe = registry.get_arbitrage_universe()

    # Symbols always open (crypto) – replaces suggest_always_open()
    always_open = registry.get_always_open()

    # Full flat list across all categories
    all_symbols = registry.get_all()
"""

from __future__ import annotations

import threading
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Minimal fallback lists — used ONLY when the broker is disconnected.
# These are conservative: better to trade fewer symbols than to assume
# availability. Update them only when genuinely needed for offline testing.
# ---------------------------------------------------------------------------

_FALLBACK: dict[str, list[str]] = {
    "forex":       ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
                    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"],
    "metals":      ["XAUUSD", "XAGUSD"],
    "crypto":      ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"],
    "indices":     ["US30", "US500", "NAS100", "GER40", "UK100"],
    "commodities": ["USOIL", "NGAS"],
    "stocks":      ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL"]
    
}

# Spread ceiling (pips) used to filter out illiquid symbols from the
# arbitrage universe. Symbols with a wider spread than this are skipped.
_ARB_MAX_SPREAD_PIPS: dict[str, float] = {
    "forex":       3.0,
    "metals":      5.0,
    "crypto":      80.0,
    "indices":     5.0,
    "commodities": 8.0,
    "stocks":      10.0
}

# How many seconds cached data stays fresh before the next broker query.
_TTL_SECONDS = 300   # 5 minutes


class SymbolRegistry:
    """
    Broker-aware, thread-safe symbol catalogue.

    The registry is lazy: it does not query the broker until the first
    call to any get_*() method, so constructing it in __init__ is free.

    Parameters
    ----------
    broker : Trader
        The connected (or disconnected) Trader instance.  When disconnected,
        all queries fall back to _FALLBACK automatically.
    ttl_seconds : int
        How long a per-category cache entry stays valid before being
        re-fetched from the broker.  Default: 300 s (5 min).
    """

    def __init__(self, broker, ttl_seconds: int = _TTL_SECONDS):
        self._broker = broker
        self._ttl    = ttl_seconds
        self._lock   = threading.RLock()

        # {category: (fetch_timestamp, [symbol_name, ...])}
        self._cache: dict[str, tuple[float, list[str]]] = {}

        # Flat set of all known symbols (union of all categories).
        # Populated lazily and refreshed whenever any category is refreshed.
        self._all_known: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_universe(self, category: Optional[str] = None) -> list[str]:
        """
        Return all broker-available symbols, optionally filtered by category.

        Parameters
        ----------
        category : str or None
            One of: "forex", "metals", "crypto", "indices", "commodities".
            Pass None to get the full cross-category list.
        """
        if category is not None:
            return self._get_category(category)

        result: list[str] = []
        for cat in _FALLBACK:
            result.extend(self._get_category(cat))
        return result

    def get_all(self) -> list[str]:
        """Flat list of every known symbol across all categories."""
        return self.get_universe()

    def get_always_open(self) -> list[str]:
        """
        Return crypto symbols available 24/7.
        Replaces the hardcoded list in MarketSessionManager.suggest_always_open().
        """
        crypto = self._get_category("crypto")
        # Prefer well-known, liquid pairs first
        priority = ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "ADAUSD"]
        ordered  = [s for s in priority if s in crypto]
        ordered += [s for s in crypto if s not in priority]
        return ordered

    def get_arbitrage_universe(self, max_per_category: int = 20) -> list[str]:
        """
        Return the best candidates for the arbitrage engine.

        Criteria:
        - Liquid categories only (forex, metals, crypto, indices)
        - Spread ≤ category ceiling (filters illiquid exotics)
        - Capped at max_per_category per bucket so the correlation matrix
          stays tractable
        """
        if not self._broker.connected:
            # Disconnected: combine fallback forex + metals as safe default
            return _FALLBACK["forex"][:max_per_category] + _FALLBACK["metals"]

        liquid_cats = ["forex", "metals", "crypto", "indices"]
        result: list[str] = []

        for cat in liquid_cats:
            ceiling = _ARB_MAX_SPREAD_PIPS.get(cat, 5.0)
            # Ask the broker for symbols with live spread information
            try:
                raw = self._broker.search_symbols(category=cat, max_results=60)
            except Exception:
                raw = []

            # Filter by spread and take the tightest max_per_category
            liquid = [
                r["name"] for r in raw
                if r.get("spread_pips") is not None
                and r["spread_pips"] <= ceiling
            ]
            # search_symbols already returns spread-sorted results on many
            # brokers, but sort explicitly to be safe
            liquid = sorted(
                liquid,
                key=lambda s: next(
                    (r["spread_pips"] for r in raw if r["name"] == s), 999
                ),
            )
            result.extend(liquid[:max_per_category])

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in result:
            if s not in seen:
                seen.add(s)
                deduped.append(s)

        return deduped or _FALLBACK["forex"]

    def is_known(self, symbol: str) -> bool:
        """True if the symbol exists on the broker (or in the fallback list)."""
        _ = self.get_universe()   # ensure cache is populated
        with self._lock:
            return symbol.upper() in self._all_known

    def invalidate(self, category: Optional[str] = None):
        """
        Force a refresh on the next access.
        Call after portfolio changes or major market events.
        """
        with self._lock:
            if category is None:
                self._cache.clear()
                self._all_known.clear()
            elif category in self._cache:
                del self._cache[category]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_category(self, category: str) -> list[str]:
        """Return cached (or freshly fetched) symbols for one category."""
        with self._lock:
            entry = self._cache.get(category)
            if entry and (time.monotonic() - entry[0]) < self._ttl:
                return list(entry[1])

        # Cache miss or expired — fetch outside the lock so we don't block
        symbols = self._fetch(category)

        with self._lock:
            self._cache[category] = (time.monotonic(), symbols)
            self._all_known.update(symbols)

        return list(symbols)

    def _fetch(self, category: str) -> list[str]:
        """
        Query the broker for all symbols in a category.
        Falls back to _FALLBACK if the broker is disconnected or returns nothing.
        """
        if not self._broker.connected:
            return _FALLBACK.get(category, [])

        try:
            raw = self._broker.search_symbols(category=category, max_results=200)
            names = [r["name"] for r in raw if r.get("name")]
            return names if names else _FALLBACK.get(category, [])
        except Exception:
            return _FALLBACK.get(category, [])

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            loaded = list(self._cache.keys())
        return (
            f"SymbolRegistry("
            f"broker_connected={self._broker.connected}, "
            f"loaded_categories={loaded})"
        )