"""
manager/correlation_matrix.py — Real-Time Portfolio Correlation Engine

Calculates rolling cross-asset correlation matrices across all symbols
in the portfolio and blocks trade entries that would push portfolio-level
correlation above a configured ceiling.

Why this matters
----------------
The existing CorrelationGuard in risk_manager.py uses a currency-leg
heuristic (e.g. EURUSD and GBPUSD share "USD"). That's fast but coarse:
it misses instrument pairs with very different legs that still move
together (e.g. XAUUSD and AUDUSD during risk-off, or EURUSD and EURGBP
when EUR dominates).

This module computes actual rolling price correlations and answers:
  "If I add this position, what is my portfolio's worst-pair correlation?"

Architecture
------------
CorrelationMatrix   — computes and caches the N×N correlation matrix
PortfolioHeatCheck  — given open positions + a proposed new trade,
                      returns (allowed: bool, reason: str, worst_pair: str)

Usage (drop-in alongside existing CorrelationGuard)
----------------------------------------------------
    checker = PortfolioHeatCheck(cache, lookback=100, max_correlation=0.80)

    allowed, reason, worst = checker.check("GBPUSD", open_symbols=["EURUSD"])
    if not allowed:
        return f"Correlation block: {reason}"
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix (cached, TTL-based refresh)
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationMatrix:
    """
    Computes and caches the pairwise rolling correlation matrix for a set
    of symbols using their close prices.

    Parameters
    ----------
    lookback    : Number of bars used for the correlation calculation.
                  Smaller → more reactive; larger → more stable.
    ttl_seconds : How long the matrix stays valid before a re-compute.
    method      : 'pearson' | 'spearman' — Pearson for linear correlation,
                  Spearman for rank-order (more robust to outliers in crypto).
    """

    def __init__(
        self,
        lookback:    int   = 100,
        ttl_seconds: int   = 60,
        method:      str   = "pearson",
    ):
        self.lookback    = lookback
        self.ttl         = ttl_seconds
        self.method      = method

        self._matrix:    Optional[pd.DataFrame] = None
        self._timestamp: float = 0.0
        self._lock       = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        symbols: list[str],
        cache,                  # LocalCache instance
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Return the current N×N correlation matrix.
        Returns None if insufficient data is available.

        Parameters
        ----------
        symbols : List of symbol tickers to include.
        cache   : LocalCache — provides OHLCV data without hitting the broker.
        """
        with self._lock:
            if (not force_refresh
                    and self._matrix is not None
                    and (time.monotonic() - self._timestamp) < self.ttl
                    and set(self._matrix.columns) == set(symbols)):
                return self._matrix.copy()

        matrix = self._compute(symbols, cache)
        with self._lock:
            if matrix is not None:
                self._matrix  = matrix
                self._timestamp = time.monotonic()
        return matrix

    def pair_correlation(
        self,
        sym_a:   str,
        sym_b:   str,
        symbols: list[str],
        cache,
    ) -> Optional[float]:
        """Return the scalar correlation between two symbols."""
        matrix = self.get(symbols, cache)
        if matrix is None:
            return None
        if sym_a not in matrix.columns or sym_b not in matrix.columns:
            return None
        return float(matrix.loc[sym_a, sym_b])

    def most_correlated_with(
        self,
        symbol:  str,
        symbols: list[str],
        cache,
        exclude: Optional[list[str]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Find the symbol in `symbols` most correlated with `symbol`.
        Returns (symbol_name, correlation_value).
        """
        matrix = self.get(symbols + [symbol], cache)
        if matrix is None or symbol not in matrix.columns:
            return None, 0.0

        row = matrix.loc[symbol].drop(symbol, errors="ignore")
        if exclude:
            row = row.drop([s for s in exclude if s in row.index], errors="ignore")

        if row.empty:
            return None, 0.0

        best_sym  = row.abs().idxmax()
        best_corr = float(row[best_sym])
        return best_sym, best_corr

    def max_portfolio_correlation(
        self,
        symbols: list[str],
        cache,
    ) -> tuple[float, str]:
        """
        Return the highest absolute pairwise correlation in the current portfolio.
        Useful for monitoring overall heat.
        Returns (max_correlation, "SYM_A / SYM_B").
        """
        if len(symbols) < 2:
            return 0.0, ""

        matrix = self.get(symbols, cache)
        if matrix is None:
            return 0.0, ""

        max_corr = 0.0
        worst_pair = ""
        cols = list(matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = abs(float(matrix.iloc[i, j]))
                if c > max_corr:
                    max_corr   = c
                    worst_pair = f"{cols[i]} / {cols[j]}"

        return round(max_corr, 3), worst_pair

    # ── Internal computation ──────────────────────────────────────────────────

    def _compute(self, symbols: list[str], cache) -> Optional[pd.DataFrame]:
        """Build a returns DataFrame and compute the correlation matrix."""
        series_map: dict[str, pd.Series] = {}

        for sym in symbols:
            df = cache.get_raw_ohlcv(sym)
            if df is None or df.empty or len(df) < self.lookback:
                continue
            closes = df["close"].iloc[-self.lookback:]
            if closes.isnull().any():
                closes = closes.fillna(method="ffill")
            returns = closes.pct_change().dropna()
            if len(returns) >= max(self.lookback // 2, 30):
                series_map[sym] = returns

        if len(series_map) < 2:
            return None

        try:
            returns_df = pd.DataFrame(series_map)
            # Align on common index; fill short gaps with 0 return
            returns_df = returns_df.dropna(how="all").fillna(0.0)

            if self.method == "spearman":
                matrix = returns_df.rank().corr()
            else:
                matrix = returns_df.corr(method="pearson")

            return matrix.clip(-1.0, 1.0)

        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio heat checker — the gate used before order submission
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioHeatCheck:
    """
    Blocks new positions that would push portfolio correlation above
    a configurable ceiling.

    Parameters
    ----------
    cache          : LocalCache — provides OHLCV data.
    lookback       : Bars used for correlation calculation.
    max_correlation: Maximum allowed absolute correlation with any existing
                     open position (0.0 = uncorrelated, 1.0 = perfectly correlated).
                     Recommended: 0.70–0.85 depending on style.
    min_data_ratio : Fraction of `lookback` bars that must exist for a symbol
                     to be included. Symbols with less data are excluded.
    ttl_seconds    : Correlation matrix refresh interval.
    method         : 'pearson' or 'spearman'.
    """

    def __init__(
        self,
        cache,
        lookback:        int   = 100,
        max_correlation: float = 0.80,
        min_data_ratio:  float = 0.5,
        ttl_seconds:     int   = 60,
        method:          str   = "pearson",
    ):
        self._cache    = cache
        self._max_corr = max_correlation
        self._min_bars = max(30, int(lookback * min_data_ratio))
        self._matrix   = CorrelationMatrix(
            lookback=lookback, ttl_seconds=ttl_seconds, method=method
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        proposed_symbol: str,
        open_symbols:    list[str],
    ) -> tuple[bool, str, str]:
        """
        Decide if opening a position on `proposed_symbol` is safe
        given the currently open positions.

        Returns
        -------
        (allowed: bool, reason: str, worst_pair: str)

        allowed    : True → trade is safe, False → block it.
        reason     : Human-readable explanation.
        worst_pair : The pair causing the block (e.g. "EURUSD / GBPUSD").
        """
        if not open_symbols:
            return True, "No existing positions — no correlation risk.", ""

        all_symbols = list(dict.fromkeys(open_symbols + [proposed_symbol]))
        matrix = self._matrix.get(all_symbols, self._cache)

        if matrix is None or proposed_symbol not in matrix.columns:
            # Can't compute → allow but warn
            return True, "Correlation data unavailable — proceeding without check.", ""

        # Check correlation of proposed symbol against each open position
        worst_corr  = 0.0
        worst_pair  = ""
        blocking_pairs: list[str] = []

        for open_sym in open_symbols:
            if open_sym not in matrix.columns:
                continue
            corr = abs(float(matrix.loc[proposed_symbol, open_sym]))
            if corr > worst_corr:
                worst_corr = corr
                worst_pair = f"{proposed_symbol} / {open_sym}"
            if corr > self._max_corr:
                blocking_pairs.append(
                    f"{open_sym} (r={corr:.2f})"
                )

        if blocking_pairs:
            reason = (
                f"High correlation detected: {proposed_symbol} is strongly "
                f"correlated with {', '.join(blocking_pairs)}. "
                f"Ceiling is r={self._max_corr:.2f}. "
                f"Adding this position would amplify directional risk."
            )
            return False, reason, worst_pair

        reason = (
            f"Correlation check passed. "
            f"Highest correlation with open positions: "
            f"r={worst_corr:.2f} ({worst_pair or 'N/A'})."
        )
        return True, reason, worst_pair

    def portfolio_heat_report(self, open_symbols: list[str]) -> str:
        """
        Generate a human-readable portfolio correlation heat report.
        Useful for the ARIA 'risk check' intent.
        """
        if len(open_symbols) < 2:
            return "Only one position open — no cross-asset correlation to measure."

        matrix = self._matrix.get(open_symbols, self._cache)
        if matrix is None:
            return "Insufficient price history to compute correlation matrix."

        max_corr, worst_pair = self._matrix.max_portfolio_correlation(
            open_symbols, self._cache
        )

        lines = [
            "Portfolio Correlation Heat Report",
            "─" * 36,
        ]

        # Overall heat level
        if max_corr >= 0.90:
            heat = "🔴 CRITICAL"
        elif max_corr >= 0.75:
            heat = "🟠 HIGH"
        elif max_corr >= 0.55:
            heat = "🟡 MODERATE"
        else:
            heat = "🟢 LOW"

        lines.append(
            f"Overall heat : {heat} (max r={max_corr:.2f}, {worst_pair})"
        )
        lines.append(f"Ceiling      : r={self._max_corr:.2f}")
        lines.append("")

        # Pairwise table
        cols = list(matrix.columns)
        lines.append("Pairwise correlations:")
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(matrix.iloc[i, j])
                flag = " ⚠️" if abs(r) > self._max_corr else ""
                lines.append(f"  {cols[i]:10s} ↔ {cols[j]:10s}  r = {r:+.3f}{flag}")

        return "\n".join(lines)

    def update_max_correlation(self, new_ceiling: float) -> None:
        """Adjust the correlation ceiling at runtime (e.g. from a chat command)."""
        self._max_corr = max(0.0, min(new_ceiling, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper: cluster portfolio by correlation
# ─────────────────────────────────────────────────────────────────────────────

def cluster_by_correlation(
    symbols: list[str],
    cache,
    lookback:   int   = 100,
    threshold:  float = 0.70,
) -> list[list[str]]:
    """
    Group symbols into clusters where intra-cluster correlation ≥ threshold.
    Useful for choosing which position to hedge or scale down when heat is high.

    Returns a list of clusters (each cluster is a list of symbol names).

    Example
    -------
        clusters = cluster_by_correlation(["EURUSD","GBPUSD","XAUUSD","BTCUSD"], cache)
        # → [["EURUSD", "GBPUSD"], ["XAUUSD"], ["BTCUSD"]]
    """
    cm   = CorrelationMatrix(lookback=lookback)
    matrix = cm.get(symbols, cache)

    if matrix is None:
        # No data → each symbol is its own cluster
        return [[s] for s in symbols]

    visited: set[str] = set()
    clusters: list[list[str]] = []

    for sym in symbols:
        if sym in visited or sym not in matrix.columns:
            continue
        cluster = [sym]
        visited.add(sym)
        for other in symbols:
            if other in visited or other not in matrix.columns or other == sym:
                continue
            if abs(float(matrix.loc[sym, other])) >= threshold:
                cluster.append(other)
                visited.add(other)
        clusters.append(cluster)

    return clusters