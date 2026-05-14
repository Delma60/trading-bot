"""
strategies/symbol_strategy_affinity.py

Learns which strategy performs best on which symbol in which regime.
Drop this file into your project, then wire it into StrategyManager
following the integration notes at the bottom.

How it works
------------
1. Every time a trade closes, record_outcome() is called with
   (symbol, strategy, regime, profit).
2. The map computes a rolling win_rate and avg_profit per
   (symbol, strategy, regime) triplet.
3. check_signals() in StrategyManager calls get_top_strategies()
   to get an ordered, weighted list of strategies for this symbol
   instead of running all of them equally.
4. The map is persisted as JSON so it survives restarts.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


AFFINITY_FILE = Path("data/symbol_strategy_affinity.json")
MAX_OUTCOMES_PER_KEY = 100   # rolling window per (symbol, strategy, regime)
MIN_TRADES_TO_TRUST  = 5     # need this many before we act on the score
DEFAULT_WEIGHT       = 1.0   # for strategies with no data yet


@dataclass
class StrategyOutcome:
    timestamp: str
    profit: float
    win: bool


@dataclass
class AffinityRecord:
    """Rolling performance of one strategy on one symbol in one regime."""
    symbol:   str
    strategy: str
    regime:   str
    outcomes: list = field(default_factory=list)   # list of {profit, win, ts}

    # ── Derived stats (not persisted, computed on demand) ─────────────────
    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def win_rate(self) -> float:
        if self.n == 0:
            return 0.5   # neutral prior
        return sum(1 for o in self.outcomes if o["win"]) / self.n

    @property
    def avg_profit(self) -> float:
        if self.n == 0:
            return 0.0
        return sum(o["profit"] for o in self.outcomes) / self.n

    @property
    def confidence(self) -> float:
        """How much we trust this score. Rises with sample count."""
        return min(self.n / MIN_TRADES_TO_TRUST, 1.0)

    def score(self) -> float:
        """
        Blended affinity score for ranking strategies.
        Combines win rate, average profit sign, and sample confidence.
        Falls back to neutral when we don't have enough data.
        """
        if self.n < MIN_TRADES_TO_TRUST:
            return DEFAULT_WEIGHT   # no preference yet

        wr_component  = (self.win_rate - 0.5) * 2.0        # -1.0 to +1.0
        pnl_component = 1.0 if self.avg_profit > 0 else -0.5
        raw = 1.0 + (wr_component * 0.7 + pnl_component * 0.3)
        return max(0.1, min(raw, 3.0))   # clamp to [0.1, 3.0]


class SymbolStrategyAffinityMap:
    """
    Persisted, thread-safe map of (symbol × strategy × regime) → performance.

    Quick start
    -----------
        affinity = SymbolStrategyAffinityMap()

        # After a trade closes:
        affinity.record_outcome("EURUSD", "Mean_Reversion", "Ranging / Choppy", profit=12.5)

        # Before running strategies:
        ranked = affinity.get_top_strategies("EURUSD", "Ranging / Choppy", all_strategies)
        # Returns list of (strategy_name, weight) pairs, best first.
    """

    def __init__(self, filepath: Path = AFFINITY_FILE):
        self._path  = filepath
        self._lock  = threading.RLock()
        self._data: dict[str, AffinityRecord] = {}
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def record_outcome(
        self,
        symbol:   str,
        strategy: str,
        regime:   str,
        profit:   float,
    ) -> None:
        """Call this after every trade close."""
        key = self._key(symbol, strategy, regime)
        with self._lock:
            if key not in self._data:
                self._data[key] = AffinityRecord(
                    symbol=symbol, strategy=strategy, regime=regime
                )
            rec = self._data[key]
            rec.outcomes.append({
                "profit": profit,
                "win":    profit > 0,
                "ts":     datetime.now().isoformat(),
            })
            # Rolling window — drop oldest if over limit
            if len(rec.outcomes) > MAX_OUTCOMES_PER_KEY:
                rec.outcomes = rec.outcomes[-MAX_OUTCOMES_PER_KEY:]
            self._save()

    def get_top_strategies(
        self,
        symbol:       str,
        regime:       str,
        all_strategies: list[str],
        top_n:        int  = 3,
        min_weight:   float = 0.1,
    ) -> list[tuple[str, float]]:
        """
        Return (strategy, weight) pairs for this symbol+regime, best first.

        Strategies with no data yet are included at DEFAULT_WEIGHT so they
        still get a chance to collect samples (exploration).

        If top_n == -1, all strategies are returned (full ensemble, weighted).
        """
        with self._lock:
            scored = []
            for strat in all_strategies:
                key = self._key(symbol, strat, regime)
                rec = self._data.get(key)
                w   = rec.score() if rec else DEFAULT_WEIGHT
                scored.append((strat, w))

        # Sort descending by weight
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_n == -1:
            return [(s, w) for s, w in scored if w >= min_weight]
        return [(s, w) for s, w in scored[:top_n] if w >= min_weight]

    def get_weight(self, symbol: str, strategy: str, regime: str) -> float:
        """Single weight lookup — used to scale confidence scores."""
        with self._lock:
            key = self._key(symbol, strategy, regime)
            rec = self._data.get(key)
            return rec.score() if rec else DEFAULT_WEIGHT

    def summary(self, symbol: str = None, top_n: int = 10) -> str:
        """Human-readable leaderboard for ARIA to surface."""
        with self._lock:
            records = list(self._data.values())

        if symbol:
            records = [r for r in records if r.symbol == symbol]

        records.sort(key=lambda r: r.score(), reverse=True)
        lines = [f"Strategy affinity{' for ' + symbol if symbol else ''}:"]
        for r in records[:top_n]:
            bar = "▓" * int(r.win_rate * 10) + "░" * (10 - int(r.win_rate * 10))
            lines.append(
                f"  {r.strategy:<22} {r.symbol:<10} [{r.regime[:18]:<18}] "
                f"WR {r.win_rate:.0%} [{bar}] n={r.n:>3}  score={r.score():.2f}"
            )
        if not records:
            lines.append("  No data yet.")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._data.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for k, v in raw.items():
                self._data[k] = AffinityRecord(
                    symbol   = v["symbol"],
                    strategy = v["strategy"],
                    regime   = v["regime"],
                    outcomes = v.get("outcomes", []),
                )
        except Exception:
            self._data = {}

    @staticmethod
    def _key(symbol: str, strategy: str, regime: str) -> str:
        return f"{symbol}|{strategy}|{regime}"


# ── Integration patches for StrategyManager ────────────────────────────────
#
# 1. In StrategyManager.__init__, add:
#
      
#
# 2. In StrategyManager.check_signals(), replace the fixed ensemble loop:
#
#   OLD:
#       for name in self.active_ensemble_strategies:
#           strategy_signals[name] = engine.analyze(...)
#
#   NEW (weight-filtered ensemble):
      
#
# 3. In StrategyManager.record_trade_outcome(), add:
#
#
#    But you need the symbol here — update the call signature:
#       def record_trade_outcome(self, feature_vector, action, profit,
#                                symbol="", strategy="", regime=""):
#
# 4. In ARIA._on_external_close(), after the episodic_memory.store() call, add:
#
      
# 5. In MetaScorer._weighted_vote(), multiply each strategy's vote weight
#    by the affinity_weight attached to the signal dict:
#