"""
strategies/arbitrage.py — Statistical Arbitrage (Pairs Trading)

Dynamically finds highly correlated assets and bets on mean-reversion
when their historical spread diverges via Z-Score.

Changes from v1
---------------
- Removed hardcoded `market_universe` list.
- The correlation search universe now comes from SymbolRegistry, which
  queries the connected broker at runtime.  This means:
    * Any pair the broker offers can become a sister pair.
    * The universe stays current without code changes when the broker
      adds or removes instruments.
- SymbolRegistry is injected at construction time so the strategy
  stays side-effect-free and independently testable.
- Fallback: if no registry is provided (e.g. unit tests), a small
  conservative default list is used — not the full broker catalogue.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# Conservative fallback used only when no SymbolRegistry is injected.
_FALLBACK_UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "US30", "NAS100",
]


class ArbitrageStrategy:
    """
    Statistical Arbitrage (Pairs Trading).

    On each call to analyze():
    1. Resolves the search universe from the injected SymbolRegistry
       (or falls back to _FALLBACK_UNIVERSE if none was provided).
    2. Finds the most correlated asset to the primary symbol using
       D1 close correlations (result is cached per primary symbol).
    3. Computes the Z-Score of the price-ratio spread on H1 data.
    4. Signals BUY when the primary is statistically cheap vs the pair,
       SELL when statistically expensive.

    Parameters
    ----------
    symbol_registry : SymbolRegistry | None
        Injected at construction time by StrategyManager.  When None,
        the strategy falls back to the small built-in universe.
    z_score_threshold : float
        How many standard deviations beyond the mean triggers a signal.
    lookback : int
        Rolling window (bars) for Z-Score calculation.
    min_correlation : float
        Pearson r threshold; candidates below this are rejected.
    """

    def __init__(
        self,
        symbol_registry=None,
        z_score_threshold: float = 2.0,
        lookback: int = 20,
        min_correlation: float = 0.80,
    ):
        self._registry          = symbol_registry   # SymbolRegistry or None
        self.z_score_threshold  = z_score_threshold
        self.lookback           = lookback
        self.min_correlation    = min_correlation

        # {primary_symbol: sister_symbol | None}
        # Cached so we don't re-run the correlation matrix every bar.
        self._pair_cache: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Universe resolution
    # ------------------------------------------------------------------

    def _universe(self) -> list[str]:
        """
        Return the set of symbols to search for a correlated pair.

        Uses the SymbolRegistry when available (broker-sourced, live),
        otherwise falls back to the hardcoded list.
        """
        if self._registry is not None:
            try:
                return self._registry.get_arbitrage_universe()
            except Exception:
                pass
        return _FALLBACK_UNIVERSE

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    def _find_correlated_pair(
        self, symbol: str, broker
    ) -> Optional[str]:
        """
        Scan the universe to find the asset most correlated with `symbol`.

        The result is cached so the expensive Daily-data correlation
        matrix is only computed once per primary symbol per session.
        """
        if symbol in self._pair_cache:
            return self._pair_cache[symbol]

        primary_df = broker.get_historical_rates(symbol, timeframe="D1", count=100)
        if primary_df is None or primary_df.empty:
            self._pair_cache[symbol] = None
            return None

        universe = self._universe()
        best_pair: Optional[str] = None
        highest_correlation = 0.0

        for candidate in universe:
            if candidate == symbol:
                continue

            candidate_df = broker.get_historical_rates(
                candidate, timeframe="D1", count=100
            )
            if candidate_df is None or candidate_df.empty:
                continue

            min_len = min(len(primary_df), len(candidate_df))
            series_a = primary_df["close"].iloc[-min_len:]
            series_b = candidate_df["close"].iloc[-min_len:]

            corr = series_a.corr(series_b)
            if pd.isna(corr):
                continue

            if corr > highest_correlation and corr >= self.min_correlation:
                highest_correlation = corr
                best_pair = candidate

        self._pair_cache[symbol] = best_pair
        return best_pair

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, symbol: str, broker) -> dict:
        """
        Compute the Z-Score of the spread between `symbol` and its most
        correlated pair and emit a directional signal.

        Parameters
        ----------
        df : pd.DataFrame
            H1 OHLCV data for `symbol` (at least lookback + 10 bars).
        symbol : str
            Primary instrument ticker.
        broker : Trader
            Connected broker used to fetch the sister pair's data.

        Returns
        -------
        dict with keys: action, confidence, reason
        """
        sister_symbol = self._find_correlated_pair(symbol, broker)
        if not sister_symbol:
            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason":     f"No correlated pair found for {symbol} in "
                              f"the broker universe ({len(self._universe())} symbols checked).",
            }

        sister_df = broker.get_historical_rates(
            sister_symbol, timeframe="H1", count=self.lookback + 10
        )

        if sister_df is None or sister_df.empty or len(df) < self.lookback:
            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason":     f"Could not fetch H1 data for sister pair {sister_symbol}.",
            }

        # ── Align and compute spread ───────────────────────────────────
        min_len = min(len(df), len(sister_df))
        asset_a = df["close"].iloc[-min_len:].values
        asset_b = sister_df["close"].iloc[-min_len:].values

        spread = asset_a / asset_b
        spread_s = pd.Series(spread)

        rolling_mean = spread_s.rolling(window=self.lookback).mean()
        rolling_std  = spread_s.rolling(window=self.lookback).std()

        current_spread = float(spread_s.iloc[-1])
        current_mean   = float(rolling_mean.iloc[-1])
        current_std    = float(rolling_std.iloc[-1])

        if pd.isna(current_std) or current_std == 0:
            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason":     "Insufficient volatility to calculate Z-Score.",
            }

        z_score = (current_spread - current_mean) / current_std

        # ── Signal rules ──────────────────────────────────────────────
        if z_score < -self.z_score_threshold:
            confidence = min(
                0.99,
                0.75 + (abs(z_score) - self.z_score_threshold) * 0.1,
            )
            return {
                "action":     "BUY",
                "confidence": round(confidence, 2),
                "reason": (
                    f"Stat-Arb: {symbol} undervalued vs {sister_symbol} "
                    f"(Z: {z_score:.2f}, pair from broker universe of "
                    f"{len(self._universe())} symbols)"
                ),
            }

        if z_score > self.z_score_threshold:
            confidence = min(
                0.99,
                0.75 + (z_score - self.z_score_threshold) * 0.1,
            )
            return {
                "action":     "SELL",
                "confidence": round(confidence, 2),
                "reason": (
                    f"Stat-Arb: {symbol} overvalued vs {sister_symbol} "
                    f"(Z: {z_score:.2f}, pair from broker universe of "
                    f"{len(self._universe())} symbols)"
                ),
            }

        return {
            "action":     "WAIT",
            "confidence": 0.0,
            "reason":     f"Spread is normal vs {sister_symbol} (Z: {z_score:.2f}).",
        }