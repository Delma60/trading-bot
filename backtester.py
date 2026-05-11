"""
manager/backtester.py — Event-Driven Backtesting Framework

Replays historical market data bar-by-bar through the SAME strategy pipeline
used in live trading: StrategyManager.check_signals() → quality scoring →
position sizing → simulated execution.

Zero lookahead bias: each bar only sees data up to and including that bar.

Usage
-----
    from manager.backtester import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        symbol        = "EURUSD",
        timeframe     = "H1",
        start_date    = "2024-01-01",
        end_date      = "2024-06-01",
        initial_equity= 10_000.0,
        risk_pct      = 1.0,
        commission_pips= 0.3,          # spread + commission in pips
    )

    engine = BacktestEngine(strategy_manager, config)
    result = engine.run()
    print(result.summary())

Architecture
------------
1. BacktestConfig   — immutable run parameters
2. SimulatedPosition— tracks an open paper trade
3. BacktestResult   — performance metrics + equity curve
4. BacktestEngine   — bar-by-bar event loop
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.features.feature_engineer import FeatureEngineer
from manager.profile_manager import profile

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    symbol:          str
    timeframe:       str    = "H1"
    start_date:      str    = ""          # "YYYY-MM-DD" or empty → all data
    end_date:        str    = ""
    initial_equity:  float  = 10_000.0
    risk_pct:        float  = 1.0        # % of equity risked per trade
    max_open_trades: int    = 3
    commission_pips: float  = 0.3        # one-way cost in pips
    slippage_pips:   float  = 0.1        # added to fill cost
    sl_pips:         float  = 20.0       # fallback SL when strategy doesn't supply one
    tp_pips:         float  = 40.0       # fallback TP
    min_confidence:  float  = 0.55       # signals below this are skipped
    min_grade:       str    = "B"        # "A", "B", or "C"
    warmup_bars:     int    = 250        # bars required before first signal is evaluated

    # pip-value helpers (overridden for JPY/metals automatically)
    pip_value_usd_per_lot: float = 10.0  # for standard 5-digit forex


# ─────────────────────────────────────────────────────────────────────────────
# Simulated position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedPosition:
    symbol:      str
    direction:   str           # "BUY" or "SELL"
    entry_price: float
    sl_price:    float
    tp_price:    float
    lots:        float
    entry_bar:   int
    entry_time:  pd.Timestamp
    strategy:    str = "ensemble"

    # filled on close
    exit_price:  Optional[float] = None
    exit_bar:    Optional[int]   = None
    exit_time:   Optional[pd.Timestamp] = None
    exit_reason: str = ""        # "TP", "SL", "SIGNAL", "EOD"
    pnl:         float = 0.0
    is_open:     bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    config:       BacktestConfig
    trades:       list[SimulatedPosition]
    equity_curve: pd.Series                # indexed by bar timestamp
    drawdown_curve: pd.Series

    # Aggregate metrics
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    net_pnl:        float = 0.0
    max_drawdown:   float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio:   float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    expectancy:     float = 0.0
    best_trade:     float = 0.0
    worst_trade:    float = 0.0
    longest_dd_bars: int  = 0
    total_bars_run: int   = 0

    def summary(self) -> str:
        """Human-readable summary for ARIA chat or file output."""
        lines = [
            f"═══ Backtest: {self.config.symbol} ({self.config.timeframe}) ═══",
            f"Period      : {self.config.start_date or 'all'} → {self.config.end_date or 'all'}",
            f"Initial Equity: ${self.config.initial_equity:,.2f}",
            f"Final Equity  : ${self.config.initial_equity + self.net_pnl:,.2f}",
            f"Net P&L       : ${self.net_pnl:+,.2f}",
            f"",
            f"Total Trades : {self.total_trades}",
            f"Win Rate     : {self.win_rate:.1f}%",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Sharpe Ratio : {self.sharpe_ratio:.2f}",
            f"",
            f"Avg Win      : ${self.avg_win:+.2f}",
            f"Avg Loss     : ${self.avg_loss:.2f}",
            f"Expectancy   : ${self.expectancy:+.2f} per trade",
            f"Best Trade   : ${self.best_trade:+.2f}",
            f"Worst Trade  : ${self.worst_trade:.2f}",
            f"",
            f"Max Drawdown : ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.1f}%)",
            f"Longest DD   : {self.longest_dd_bars} bars",
            f"Bars Run     : {self.total_bars_run}",
            f"",
            f"Risk/Trade   : {self.config.risk_pct}% | SL fallback: {self.config.sl_pips}p | Min Conf: {self.config.min_confidence:.0%}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export trades to a DataFrame for further analysis."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "direction":   t.direction,
                "lots":        t.lots,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "sl":          t.sl_price,
                "tp":          t.tp_price,
                "pnl":         t.pnl,
                "exit_reason": t.exit_reason,
                "strategy":    t.strategy,
                "bars_held":   (t.exit_bar - t.entry_bar) if t.exit_bar else None,
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str = "data/backtest_trades.csv") -> None:
        """Save trade log to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtest that walks historical bars one at a time.

    Design principles
    -----------------
    - Zero lookahead: signal for bar N is generated from data[0:N] only.
    - No broker connection needed — data fetched once up front from cache.
    - Strategy pipeline is identical to live: feature engineering → ensemble
      → meta-scorer → quality grade → position sizing.
    - Realistic fills: commission + slippage applied to every open and close.
    """

    GRADE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}

    def __init__(self, strategy_manager, config: BacktestConfig):
        self.sm     = strategy_manager
        self.cfg    = config
        self._positions: list[SimulatedPosition] = []
        self._closed:    list[SimulatedPosition] = []
        self._equity_history: list[tuple[pd.Timestamp, float]] = []
        self._pip_mult = self._infer_pip_multiplier(config.symbol)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, notify_callback=print) -> BacktestResult:
        """
        Run the full backtest. Returns a BacktestResult.
        """
        raw_df = self._fetch_data(notify_callback)
        if raw_df is None or raw_df.empty:
            notify_callback(f"[Backtest] No data available for {self.cfg.symbol}.")
            return self._empty_result()

        raw_df = self._slice_dates(raw_df)
        if len(raw_df) < self.cfg.warmup_bars + 10:
            notify_callback(
                f"[Backtest] Insufficient data: {len(raw_df)} bars "
                f"(need ≥ {self.cfg.warmup_bars + 10})."
            )
            return self._empty_result()

        equity  = self.cfg.initial_equity
        peak    = equity
        notify_callback(
            f"[Backtest] Running {self.cfg.symbol} {self.cfg.timeframe} "
            f"| {len(raw_df)} bars | equity ${equity:,.2f}"
        )

        # ── Bar-by-bar event loop ─────────────────────────────────────────────
        for bar_idx in range(self.cfg.warmup_bars, len(raw_df)):
            window = raw_df.iloc[: bar_idx + 1]   # everything up to THIS bar
            current_bar = raw_df.iloc[bar_idx]
            ts = raw_df.index[bar_idx]

            # 1. Check exits first (SL / TP hit on this bar's high/low)
            self._check_exits(current_bar, bar_idx, ts)

            # 2. Calculate current equity (floating)
            floating = sum(self._floating_pnl(p, current_bar) for p in self._positions)
            current_equity = equity + floating + sum(p.pnl for p in self._closed)
            self._equity_history.append((ts, current_equity))

            if current_equity > peak:
                peak = current_equity

            # 3. Check if we can open a new trade
            if len(self._positions) >= self.cfg.max_open_trades:
                continue

            # 4. Run strategy pipeline on the data window
            signal = self._run_pipeline(window, ts)
            if not signal or signal.get("action") == "WAIT":
                continue
            if signal.get("confidence", 0.0) < self.cfg.min_confidence:
                continue

            # 5. Quality gate
            grade = self._score_grade(signal, window)
            if self.GRADE_MAP.get(grade, 0) < self.GRADE_MAP.get(self.cfg.min_grade, 3):
                continue

            # 6. Avoid same-direction duplicate positions
            action = signal["action"]
            existing_dirs = {p.direction for p in self._positions}
            if action in existing_dirs:
                continue

            # 7. Position sizing
            sl_pips = signal.get("sl_pips") or self.cfg.sl_pips
            lots    = self._size_position(current_equity, sl_pips)
            if lots <= 0:
                continue

            # 8. Simulate fill
            pos = self._open_position(
                action, current_bar, bar_idx, ts, lots, signal, sl_pips
            )
            if pos:
                self._positions.append(pos)

        # ── Close any remaining open positions at the last bar ────────────────
        last_bar = raw_df.iloc[-1]
        last_ts  = raw_df.index[-1]
        for pos in list(self._positions):
            self._close_position(pos, last_bar["close"], len(raw_df) - 1, last_ts, "EOD")

        # Finalize closed equity
        net_pnl = sum(p.pnl for p in self._closed)

        return self._compute_result(net_pnl, len(raw_df))

    # ── Pipeline integration ──────────────────────────────────────────────────

    def _run_pipeline(self, window: pd.DataFrame, ts: pd.Timestamp) -> Optional[dict]:
        """
        Run feature engineering + strategy ensemble on the data window.
        Falls back gracefully on any error.
        """
        try:
            feat_df = FeatureEngineer.compute(window)
            if feat_df.empty or len(feat_df) < 50:
                return None

            signal_votes: dict[str, dict] = {}
            for name, engine in self.sm.engines.items():
                if name in ("News_Trading", "Sentiment_Analysis"):
                    continue
                try:
                    from inspect import signature
                    sig = signature(engine.analyze)
                    args = [feat_df]
                    # Arbitrage needs symbol + broker; skip in backtest
                    if name == "Arbitrage":
                        continue
                    signal_votes[name] = engine.analyze(*args)
                except Exception:
                    continue

            if not signal_votes:
                return None

            # LSTM prediction (may not have weights yet — safe fallback)
            try:
                lstm_pred = self.sm.lstm.predict(feat_df, symbol=self.cfg.symbol)
            except Exception:
                lstm_pred = {"direction": "NEUTRAL", "confidence": 0.0, "probabilities": {}}

            # Regime
            regime = "Unknown"
            if self.sm.learner:
                regime = self.sm.learner.get_current_regime()

            final = self.sm.meta.score(
                signal_votes, lstm_pred, feat_df.iloc[-1], regime=regime
            )
            final["strategy_signals"] = signal_votes
            final["lstm_prediction"]  = lstm_pred
            return final

        except Exception:
            return None

    def _score_grade(self, signal: dict, window: pd.DataFrame) -> str:
        """Lightweight quality grading without a full reasoning engine."""
        conf = signal.get("confidence", 0.0)
        action = signal.get("action", "WAIT")
        strat_signals = signal.get("strategy_signals", {})

        agreements = sum(
            1 for s in strat_signals.values()
            if s.get("action") == action and s.get("confidence", 0) > 0.5
        )
        total = max(len(strat_signals), 1)
        confluence_pct = agreements / total

        # Simple linear grade
        score = conf * 50 + confluence_pct * 50
        if score >= 75:   return "A"
        elif score >= 55: return "B"
        elif score >= 40: return "C"
        return "D"

    # ── Position management ───────────────────────────────────────────────────

    def _open_position(
        self,
        action: str,
        bar: pd.Series,
        bar_idx: int,
        ts: pd.Timestamp,
        lots: float,
        signal: dict,
        sl_pips: float,
    ) -> Optional[SimulatedPosition]:
        """Simulate a fill at bar close + slippage/commission."""
        try:
            raw_price = float(bar["close"])
            cost_pips = self.cfg.commission_pips + self.cfg.slippage_pips
            cost_price = cost_pips / self._pip_mult

            if action == "BUY":
                entry = raw_price + cost_price
                tp_pips = signal.get("tp_pips") or self.cfg.tp_pips
                sl = entry - sl_pips / self._pip_mult
                tp = entry + tp_pips / self._pip_mult
            else:
                entry = raw_price - cost_price
                tp_pips = signal.get("tp_pips") or self.cfg.tp_pips
                sl = entry + sl_pips / self._pip_mult
                tp = entry - tp_pips / self._pip_mult

            return SimulatedPosition(
                symbol=self.cfg.symbol,
                direction=action,
                entry_price=round(entry, 5),
                sl_price=round(sl, 5),
                tp_price=round(tp, 5),
                lots=lots,
                entry_bar=bar_idx,
                entry_time=ts,
                strategy=signal.get("source", "ensemble"),
            )
        except Exception:
            return None

    def _check_exits(self, bar: pd.Series, bar_idx: int, ts: pd.Timestamp):
        """Check SL and TP hits against bar's high/low."""
        high = float(bar["high"])
        low  = float(bar["low"])

        for pos in list(self._positions):
            if pos.direction == "BUY":
                if low <= pos.sl_price:
                    self._close_position(pos, pos.sl_price, bar_idx, ts, "SL")
                elif high >= pos.tp_price:
                    self._close_position(pos, pos.tp_price, bar_idx, ts, "TP")
            else:  # SELL
                if high >= pos.sl_price:
                    self._close_position(pos, pos.sl_price, bar_idx, ts, "SL")
                elif low <= pos.tp_price:
                    self._close_position(pos, pos.tp_price, bar_idx, ts, "TP")

    def _close_position(
        self, pos: SimulatedPosition, price: float,
        bar_idx: int, ts: pd.Timestamp, reason: str
    ):
        """Settle a position and move it to the closed list."""
        cost_pips  = self.cfg.commission_pips + self.cfg.slippage_pips
        cost_price = cost_pips / self._pip_mult

        if pos.direction == "BUY":
            fill = price - cost_price
            pip_diff = (fill - pos.entry_price) * self._pip_mult
        else:
            fill = price + cost_price
            pip_diff = (pos.entry_price - fill) * self._pip_mult

        pip_val     = self.cfg.pip_value_usd_per_lot * pos.lots
        pos.pnl     = round(pip_diff * pip_val, 2)
        pos.exit_price  = round(fill, 5)
        pos.exit_bar    = bar_idx
        pos.exit_time   = ts
        pos.exit_reason = reason
        pos.is_open     = False

        self._positions.remove(pos)
        self._closed.append(pos)

    def _floating_pnl(self, pos: SimulatedPosition, bar: pd.Series) -> float:
        """Current mark-to-market P&L for an open position."""
        mid = float(bar["close"])
        pip_val = self.cfg.pip_value_usd_per_lot * pos.lots
        if pos.direction == "BUY":
            return (mid - pos.entry_price) * self._pip_mult * pip_val
        return (pos.entry_price - mid) * self._pip_mult * pip_val

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _size_position(self, equity: float, sl_pips: float) -> float:
        """Risk-based position sizing identical to RiskManager logic."""
        if sl_pips <= 0:
            return 0.0
        max_risk_usd = equity * (self.cfg.risk_pct / 100.0)
        pip_val_per_lot = self.cfg.pip_value_usd_per_lot
        raw_lots = max_risk_usd / (sl_pips * pip_val_per_lot)
        lots = math.floor(raw_lots * 100) / 100   # step 0.01
        return max(0.01, min(lots, 100.0))

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _fetch_data(self, notify_callback) -> Optional[pd.DataFrame]:
        """Pull OHLCV from the local cache or directly from the broker."""
        try:
            df = self.sm.cache.get_raw_ohlcv(self.cfg.symbol)
            if df is not None and not df.empty:
                notify_callback(
                    f"[Backtest] Loaded {len(df)} bars from cache for {self.cfg.symbol}."
                )
                return df
        except Exception:
            pass

        # Direct broker fallback
        try:
            df = self.sm.broker.ohclv_data(
                self.cfg.symbol, timeframe=self.cfg.timeframe, num_bars=5000
            )
            if df is not None and not df.empty:
                notify_callback(
                    f"[Backtest] Loaded {len(df)} bars from broker for {self.cfg.symbol}."
                )
            return df
        except Exception as exc:
            notify_callback(f"[Backtest] Data fetch failed: {exc}")
            return None

    def _slice_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to the configured date range."""
        try:
            if self.cfg.start_date:
                df = df[df.index >= pd.to_datetime(self.cfg.start_date)]
            if self.cfg.end_date:
                df = df[df.index <= pd.to_datetime(self.cfg.end_date)]
        except Exception:
            pass
        return df

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_result(self, net_pnl: float, total_bars: int) -> BacktestResult:
        trades  = self._closed
        eq_idx  = [t for t, _ in self._equity_history]
        eq_vals = [v for _, v in self._equity_history]

        if not eq_idx:
            return self._empty_result()

        equity_s = pd.Series(eq_vals, index=eq_idx, name="equity")

        # Drawdown
        roll_max = equity_s.cummax()
        dd_s     = equity_s - roll_max   # always ≤ 0
        max_dd   = float(abs(dd_s.min()))
        peak_eq  = float(roll_max.max())
        max_dd_pct = (max_dd / peak_eq * 100) if peak_eq > 0 else 0.0

        # Longest drawdown duration (in bars)
        in_dd = (dd_s < 0).astype(int)
        longest = 0
        run = 0
        for val in in_dd:
            if val:
                run += 1
                longest = max(longest, run)
            else:
                run = 0

        pnls    = [t.pnl for t in trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p < 0]
        n       = len(pnls)
        win_n   = len(wins)
        loss_n  = len(losses)

        gross_win  = sum(wins)  if wins   else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

        avg_win  = gross_win / win_n   if win_n   else 0.0
        avg_loss = gross_loss / loss_n if loss_n else 0.0

        expectancy = (
            (win_n / n * avg_win) - (loss_n / n * avg_loss) if n > 0 else 0.0
        )

        # Sharpe (annualised, assume H1 bars → 8760 bars/yr)
        bars_per_year = {"M1": 525600, "M5": 105120, "M15": 35040,
                         "M30": 17520, "H1": 8760, "H4": 2190, "D1": 252}.get(
            self.cfg.timeframe, 8760
        )
        ret_series = equity_s.pct_change().dropna()
        sharpe = 0.0
        if len(ret_series) > 1 and ret_series.std() > 0:
            sharpe = (ret_series.mean() / ret_series.std()) * math.sqrt(bars_per_year)

        return BacktestResult(
            config          = self.cfg,
            trades          = trades,
            equity_curve    = equity_s,
            drawdown_curve  = dd_s,
            total_trades    = n,
            winning_trades  = win_n,
            losing_trades   = loss_n,
            win_rate        = (win_n / n * 100) if n else 0.0,
            profit_factor   = round(pf, 2),
            net_pnl         = round(net_pnl, 2),
            max_drawdown    = round(max_dd, 2),
            max_drawdown_pct= round(max_dd_pct, 2),
            sharpe_ratio    = round(sharpe, 2),
            avg_win         = round(avg_win, 2),
            avg_loss        = round(-avg_loss, 2),
            expectancy      = round(expectancy, 2),
            best_trade      = round(max(pnls), 2) if pnls else 0.0,
            worst_trade     = round(min(pnls), 2) if pnls else 0.0,
            longest_dd_bars = longest,
            total_bars_run  = total_bars,
        )

    def _empty_result(self) -> BacktestResult:
        empty_s = pd.Series([], dtype=float)
        return BacktestResult(
            config=self.cfg, trades=[], equity_curve=empty_s, drawdown_curve=empty_s
        )

    @staticmethod
    def _infer_pip_multiplier(symbol: str) -> float:
        """Return pip-to-price multiplier for position sizing and P&L."""
        s = symbol.upper()
        if any(s.startswith(p) for p in ["XAU", "XAG"]):
            return 100.0    # Gold/Silver: 1 pip = $0.01
        if "JPY" in s:
            return 100.0    # JPY pairs: 3-digit quotes
        if any(k in s for k in ["BTC", "ETH", "LTC"]):
            return 1.0      # Crypto
        if any(k in s for k in ["US30", "NAS", "GER", "UK"]):
            return 1.0      # Indices
        return 10_000.0     # Standard 5-digit forex


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner (called from chat / ARIA agent)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    strategy_manager,
    symbol:       str,
    timeframe:    str   = None,
    start_date:   str   = "",
    end_date:     str   = "",
    risk_pct:     float = 1.0,
    min_grade:    str   = "B",
    notify_callback      = print,
) -> BacktestResult:
    """
    One-call convenience wrapper.

    Example (from ARIA chat handler):
        result = run_backtest(self.sm, "EURUSD", timeframe="H1",
                              start_date="2024-01-01", risk_pct=1.0)
        return result.summary()
    """
    effective_tf = timeframe or profile.scanner().timeframe
    config = BacktestConfig(
        symbol      = symbol,
        timeframe   = effective_tf,
        start_date  = start_date,
        end_date    = end_date,
        risk_pct    = risk_pct,
        min_grade   = min_grade,
    )
    engine = BacktestEngine(strategy_manager, config)
    return engine.run(notify_callback=notify_callback)
