"""
manager/correlation_matrix.py — Real-Time Portfolio Correlation Engine

FIX #14: fillna(method="ffill") removed in pandas ≥ 2.2.
         Replaced with the recommended .ffill() call throughout.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pandas as pd


class CorrelationMatrix:
    def __init__(self, lookback: int = 100, ttl_seconds: int = 60, method: str = "pearson"):
        self.lookback    = lookback
        self.ttl         = ttl_seconds
        self.method      = method
        self._matrix:    Optional[pd.DataFrame] = None
        self._timestamp: float = 0.0
        self._lock       = threading.RLock()

    def get(self, symbols: list[str], cache, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        with self._lock:
            if (not force_refresh
                    and self._matrix is not None
                    and (time.monotonic() - self._timestamp) < self.ttl
                    and set(self._matrix.columns) == set(symbols)):
                return self._matrix.copy()

        matrix = self._compute(symbols, cache)
        with self._lock:
            if matrix is not None:
                self._matrix    = matrix
                self._timestamp = time.monotonic()
        return matrix

    def pair_correlation(self, sym_a: str, sym_b: str, symbols: list[str], cache) -> Optional[float]:
        matrix = self.get(symbols, cache)
        if matrix is None or sym_a not in matrix.columns or sym_b not in matrix.columns:
            return None
        return float(matrix.loc[sym_a, sym_b])

    def most_correlated_with(self, symbol: str, symbols: list[str], cache,
                              exclude: Optional[list[str]] = None) -> tuple[Optional[str], float]:
        matrix = self.get(symbols + [symbol], cache)
        if matrix is None or symbol not in matrix.columns:
            return None, 0.0
        row = matrix.loc[symbol].drop(symbol, errors="ignore")
        if exclude:
            row = row.drop([s for s in exclude if s in row.index], errors="ignore")
        if row.empty:
            return None, 0.0
        best_sym  = row.abs().idxmax()
        return best_sym, float(row[best_sym])

    def max_portfolio_correlation(self, symbols: list[str], cache) -> tuple[float, str]:
        if len(symbols) < 2:
            return 0.0, ""
        matrix = self.get(symbols, cache)
        if matrix is None:
            return 0.0, ""
        max_corr, worst_pair = 0.0, ""
        cols = list(matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = abs(float(matrix.iloc[i, j]))
                if c > max_corr:
                    max_corr   = c
                    worst_pair = f"{cols[i]} / {cols[j]}"
        return round(max_corr, 3), worst_pair

    def _compute(self, symbols: list[str], cache) -> Optional[pd.DataFrame]:
        series_map: dict[str, pd.Series] = {}
        for sym in symbols:
            df = cache.get_raw_ohlcv(sym)
            if df is None or df.empty or len(df) < self.lookback:
                continue
            closes  = df["close"].iloc[-self.lookback:]
            closes  = closes.ffill()          # FIX #14: was fillna(method="ffill")
            returns = closes.pct_change().dropna()
            if len(returns) >= max(self.lookback // 2, 30):
                series_map[sym] = returns

        if len(series_map) < 2:
            return None

        try:
            returns_df = pd.DataFrame(series_map)
            # FIX #14: was .fillna(method="ffill") — removed in pandas 2.2
            returns_df = returns_df.dropna(how="all").ffill().fillna(0.0)

            if self.method == "spearman":
                matrix = returns_df.rank().corr()
            else:
                matrix = returns_df.corr(method="pearson")

            return matrix.clip(-1.0, 1.0)
        except Exception:
            return None


class PortfolioHeatCheck:
    def __init__(self, cache, lookback: int = 100, max_correlation: float = 0.80,
                 min_data_ratio: float = 0.5, ttl_seconds: int = 60, method: str = "pearson"):
        self._cache    = cache
        self._max_corr = max_correlation
        self._min_bars = max(30, int(lookback * min_data_ratio))
        self._matrix   = CorrelationMatrix(lookback=lookback, ttl_seconds=ttl_seconds, method=method)

    def check(self, proposed_symbol: str, open_symbols: list[str]) -> tuple[bool, str, str]:
        if not open_symbols:
            return True, "No existing positions — no correlation risk.", ""

        all_symbols = list(dict.fromkeys(open_symbols + [proposed_symbol]))
        matrix = self._matrix.get(all_symbols, self._cache)

        if matrix is None or proposed_symbol not in matrix.columns:
            return True, "Correlation data unavailable — proceeding without check.", ""

        worst_corr, worst_pair, blocking_pairs = 0.0, "", []
        for open_sym in open_symbols:
            if open_sym not in matrix.columns:
                continue
            corr = abs(float(matrix.loc[proposed_symbol, open_sym]))
            if corr > worst_corr:
                worst_corr = corr
                worst_pair = f"{proposed_symbol} / {open_sym}"
            if corr > self._max_corr:
                blocking_pairs.append(f"{open_sym} (r={corr:.2f})")

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
        if len(open_symbols) < 2:
            return "Only one position open — no cross-asset correlation to measure."
        matrix = self._matrix.get(open_symbols, self._cache)
        if matrix is None:
            return "Insufficient price history to compute correlation matrix."

        max_corr, worst_pair = self._matrix.max_portfolio_correlation(open_symbols, self._cache)
        if max_corr >= 0.90:   heat = "🔴 CRITICAL"
        elif max_corr >= 0.75: heat = "🟠 HIGH"
        elif max_corr >= 0.55: heat = "🟡 MODERATE"
        else:                  heat = "🟢 LOW"

        lines = [
            "Portfolio Correlation Heat Report",
            "─" * 36,
            f"Overall heat : {heat} (max r={max_corr:.2f}, {worst_pair})",
            f"Ceiling      : r={self._max_corr:.2f}",
            "",
            "Pairwise correlations:",
        ]
        cols = list(matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r    = float(matrix.iloc[i, j])
                flag = " ⚠️" if abs(r) > self._max_corr else ""
                lines.append(f"  {cols[i]:10s} ↔ {cols[j]:10s}  r = {r:+.3f}{flag}")
        return "\n".join(lines)

    def update_max_correlation(self, new_ceiling: float) -> None:
        self._max_corr = max(0.0, min(new_ceiling, 1.0))


def cluster_by_correlation(symbols: list[str], cache, lookback: int = 100,
                            threshold: float = 0.70) -> list[list[str]]:
    cm     = CorrelationMatrix(lookback=lookback)
    matrix = cm.get(symbols, cache)
    if matrix is None:
        return [[s] for s in symbols]

    visited: set[str]       = set()
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