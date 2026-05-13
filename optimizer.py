"""
optimizer.py — ARIA Parameter Optimization Suite

Five optimizer classes that tune every tunable layer of the trading system
by running the existing BacktestEngine as the fitness function. Zero new
dependencies — everything is stdlib + numpy/pandas which are already in req.txt.

Classes
-------
StrategyParamOptimizer   — tunes per-strategy hyperparameters (BB, ATR, RSI…)
RiskParamOptimizer       — tunes risk_pct, SL/TP pips, drawdown thresholds
EnsembleWeightOptimizer  — tunes MetaScorer's regime-weighted vote multipliers
WalkForwardOptimizer     — validates any param set on rolling out-of-sample windows
PortfolioOptimizer       — allocates capital across symbols (Sharpe / min-DD frontier)

Integration
-----------
All optimizers accept the live strategy_manager object so they share the
already-warmed LocalCache — no extra broker calls needed during sweeps.

Quick start (from ARIA chat or a standalone script):
    from optimizer import (
        StrategyParamOptimizer, RiskParamOptimizer,
        EnsembleWeightOptimizer, WalkForwardOptimizer, PortfolioOptimizer,
        run_full_optimization,
    )

    results = run_full_optimization(
        strategy_manager, symbol="EURUSD",
        start_date="2024-01-01", end_date="2024-12-31",
        notify=print,
    )
    print(results.summary())
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtester import BacktestConfig, BacktestEngine, BacktestResult, run_backtest


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _silent(*_a, **_kw):
    pass


def _sharpe(result: BacktestResult) -> float:
    """Extract Sharpe ratio; return -inf for empty/failed runs."""
    if result is None or result.total_trades == 0:
        return -math.inf
    return result.sharpe_ratio


def _score(result: BacktestResult, metric: str = "sharpe") -> float:
    """Multi-metric fitness scorer."""
    if result is None or result.total_trades < 5:
        return -math.inf
    m = metric.lower()
    if m == "sharpe":        return result.sharpe_ratio
    if m == "profit_factor": return result.profit_factor if result.profit_factor != math.inf else 10.0
    if m == "win_rate":      return result.win_rate
    if m == "net_pnl":       return result.net_pnl
    if m == "calmar":        return (result.net_pnl / result.max_drawdown) if result.max_drawdown > 0 else 0.0
    return result.sharpe_ratio


@dataclass
class OptimizationResult:
    """Unified result container returned by every optimizer."""
    optimizer:   str
    symbol:      str
    best_params: dict
    best_metric: float
    metric_name: str
    backtest:    Optional[BacktestResult]
    all_trials:  list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    notes:       str = ""

    def summary(self) -> str:
        lines = [
            f"{'═'*55}",
            f"Optimizer    : {self.optimizer}",
            f"Symbol       : {self.symbol}",
            f"Metric       : {self.metric_name}  →  {self.best_metric:.4f}",
            f"Trials run   : {len(self.all_trials)}",
            f"Elapsed      : {self.elapsed_sec:.1f}s",
            f"Best params  :",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k:<30} {v}")
        if self.backtest:
            lines.append("")
            lines.append(self.backtest.summary())
        if self.notes:
            lines.append(f"\nNotes: {self.notes}")
        lines.append(f"{'═'*55}")
        return "\n".join(lines)

    def save(self, path: str = None) -> str:
        """Persist best params + summary to JSON."""
        out = path or f"data/optimizer_{self.optimizer}_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "optimizer":   self.optimizer,
            "symbol":      self.symbol,
            "metric":      self.metric_name,
            "best_metric": self.best_metric,
            "best_params": self.best_params,
            "elapsed_sec": self.elapsed_sec,
            "notes":       self.notes,
            "trials":      self.all_trials[-50:],   # keep last 50 to avoid huge files
        }
        Path(out).write_text(json.dumps(payload, indent=2, default=str))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. StrategyParamOptimizer
# ─────────────────────────────────────────────────────────────────────────────

# Default search spaces for each strategy.  Extend as strategies grow.
STRATEGY_PARAM_SPACES: dict[str, dict[str, list]] = {
    "Mean_Reversion": {
        "bb_length":        [14, 20, 26],
        "bb_std":           [1.5, 2.0, 2.5],
        "rsi_length":       [10, 14, 21],
        "adx_threshold":    [20.0, 25.0, 30.0],
        "min_rr":           [1.2, 1.5, 2.0],
        "min_bb_width":     [0.003, 0.005, 0.008],
    },
    "Momentum": {
        "adx_min":          [20.0, 25.0, 30.0],
        "rsi_bull_min":     [50.0, 55.0, 60.0],
        "rsi_bear_max":     [40.0, 45.0, 50.0],
        "atr_sl_mult":      [1.0, 1.5, 2.0],
        "atr_tp_base_mult": [2.0, 2.5, 3.0],
        "min_rr":           [1.2, 1.5, 2.0],
    },
    "Breakout": {
        "lookback_window":   [15, 20, 30],
        "volume_multiplier": [1.2, 1.3, 1.5],
        "atr_buffer_pct":    [0.10, 0.15, 0.20],
    },
    "Scalping": {
        "ema_fast": [3, 5, 8],
        "ema_slow": [9, 13, 21],
        "stoch_k":  [3, 5, 9],
    },
    "Trend_Following": {
        "sma_fast":       [20, 50, 100],
        "sma_slow":       [100, 200, 300],
        "adx_threshold":  [20.0, 25.0, 30.0],
        "min_cross_age":  [2, 3, 5],
    },
}


class StrategyParamOptimizer:
    """
    Tunes hyperparameters for each individual strategy using the backtest engine.

    Methods
    -------
    grid_search(strategy_name, symbol, …)   — exhaustive grid over param_space
    random_search(strategy_name, symbol, …) — random sampling (faster for large spaces)
    optimize(strategy_name, symbol, …)      — auto-selects method based on grid size
    """

    def __init__(self, strategy_manager, notify: Callable = _silent):
        self.sm     = strategy_manager
        self.notify = notify

    # ── Public API ────────────────────────────────────────────────────────────

    def grid_search(
        self,
        strategy_name: str,
        symbol:        str,
        param_space:   dict[str, list] = None,
        start_date:    str  = "",
        end_date:      str  = "",
        metric:        str  = "sharpe",
        risk_pct:      float = 1.0,
        notify:        Callable = None,
    ) -> OptimizationResult:
        """Exhaustive grid search over every combination in param_space."""
        notify = notify or self.notify
        space  = param_space or STRATEGY_PARAM_SPACES.get(strategy_name, {})
        if not space:
            return self._empty(strategy_name, symbol, "No param space defined.")

        keys   = list(space.keys())
        combos = list(itertools.product(*[space[k] for k in keys]))
        notify(f"[StrategyOpt] Grid search: {strategy_name} | {symbol} | {len(combos)} combos")

        return self._run_sweep(
            strategy_name, symbol, keys, combos,
            start_date, end_date, metric, risk_pct, notify,
        )

    def random_search(
        self,
        strategy_name: str,
        symbol:        str,
        param_space:   dict[str, list] = None,
        n_trials:      int  = 30,
        start_date:    str  = "",
        end_date:      str  = "",
        metric:        str  = "sharpe",
        risk_pct:      float = 1.0,
        notify:        Callable = None,
    ) -> OptimizationResult:
        """Random sampling — useful when the full grid would be too large."""
        notify = notify or self.notify
        space  = param_space or STRATEGY_PARAM_SPACES.get(strategy_name, {})
        if not space:
            return self._empty(strategy_name, symbol, "No param space defined.")

        keys   = list(space.keys())
        combos = [
            tuple(random.choice(space[k]) for k in keys)
            for _ in range(n_trials)
        ]
        notify(f"[StrategyOpt] Random search: {strategy_name} | {symbol} | {n_trials} trials")

        return self._run_sweep(
            strategy_name, symbol, keys, combos,
            start_date, end_date, metric, risk_pct, notify,
        )

    def optimize(
        self,
        strategy_name: str,
        symbol:        str,
        param_space:   dict[str, list] = None,
        max_trials:    int  = 50,
        start_date:    str  = "",
        end_date:      str  = "",
        metric:        str  = "sharpe",
        risk_pct:      float = 1.0,
        notify:        Callable = None,
    ) -> OptimizationResult:
        """Auto-selects grid vs random based on total grid size."""
        space = param_space or STRATEGY_PARAM_SPACES.get(strategy_name, {})
        total = 1
        for v in space.values():
            total *= len(v)

        if total <= max_trials:
            return self.grid_search(strategy_name, symbol, space,
                                    start_date, end_date, metric, risk_pct, notify)
        return self.random_search(strategy_name, symbol, space, max_trials,
                                   start_date, end_date, metric, risk_pct, notify)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_sweep(
        self,
        strategy_name: str,
        symbol:        str,
        keys:          list,
        combos:        list,
        start_date:    str,
        end_date:      str,
        metric:        str,
        risk_pct:      float,
        notify:        Callable,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        best_score  = -math.inf
        best_params = {}
        best_bt     = None
        trials      = []

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))

            # Patch the live strategy engine temporarily
            engine = self.sm.engines.get(strategy_name)
            if engine is None:
                continue

            original_attrs = {k: getattr(engine, k, None) for k in keys}
            try:
                for k, v in params.items():
                    setattr(engine, k, v)

                bt = run_backtest(
                    self.sm, symbol,
                    start_date=start_date, end_date=end_date,
                    risk_pct=risk_pct, notify_callback=_silent,
                )
                s = _score(bt, metric)
                trials.append({**params, metric: round(s, 4), "trades": bt.total_trades})

                if s > best_score:
                    best_score  = s
                    best_params = dict(params)
                    best_bt     = bt

                if i % 10 == 0:
                    notify(f"[StrategyOpt] {i}/{len(combos)} | best {metric}: {best_score:.4f}")

            finally:
                # Always restore original attrs
                for k, v in original_attrs.items():
                    if v is not None:
                        setattr(engine, k, v)

        # Apply the best params permanently on the live engine
        if best_params and (engine := self.sm.engines.get(strategy_name)):
            for k, v in best_params.items():
                setattr(engine, k, v)
            notify(f"[StrategyOpt] Best params applied to {strategy_name}: {best_params}")

        return OptimizationResult(
            optimizer   = f"StrategyParamOptimizer[{strategy_name}]",
            symbol      = symbol,
            best_params = best_params,
            best_metric = round(best_score, 4),
            metric_name = metric,
            backtest    = best_bt,
            all_trials  = trials,
            elapsed_sec = round(time.perf_counter() - t0, 2),
        )

    @staticmethod
    def _empty(strategy_name, symbol, notes) -> OptimizationResult:
        return OptimizationResult(
            optimizer=f"StrategyParamOptimizer[{strategy_name}]",
            symbol=symbol, best_params={}, best_metric=-math.inf,
            metric_name="sharpe", backtest=None, notes=notes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. RiskParamOptimizer
# ─────────────────────────────────────────────────────────────────────────────

RISK_PARAM_SPACE: dict[str, list] = {
    "risk_pct":         [0.5, 1.0, 1.5, 2.0],
    "sl_pips":          [15.0, 20.0, 25.0, 30.0, 40.0],
    "tp_pips":          [20.0, 30.0, 40.0, 60.0, 80.0],
    "atr_sl_mult":      [1.0, 1.5, 2.0],
    "atr_tp_base_mult": [2.0, 2.5, 3.0, 3.5],
}


class RiskParamOptimizer:
    """
    Tunes risk management parameters (risk_pct, SL/TP pips, drawdown thresholds).

    Works by overriding BacktestConfig fields per trial — does NOT touch the
    live RiskManager, so it's safe to run while the bot is scanning.
    """

    def __init__(self, strategy_manager, notify: Callable = _silent):
        self.sm     = strategy_manager
        self.notify = notify

    def optimize(
        self,
        symbol:      str,
        param_space: dict[str, list] = None,
        n_trials:    int   = 40,
        start_date:  str   = "",
        end_date:    str   = "",
        metric:      str   = "calmar",      # calmar = return/max-DD — great for risk tuning
        notify:      Callable = None,
    ) -> OptimizationResult:
        notify = notify or self.notify
        space  = param_space or RISK_PARAM_SPACE
        keys   = list(space.keys())
        total  = 1
        for v in space.values():
            total *= len(v)

        if total <= n_trials:
            combos = list(itertools.product(*[space[k] for k in keys]))
        else:
            combos = [tuple(random.choice(space[k]) for k in keys) for _ in range(n_trials)]

        notify(f"[RiskOpt] {symbol} | {len(combos)} trials | metric={metric}")
        t0 = time.perf_counter()

        best_score  = -math.inf
        best_params = {}
        best_bt     = None
        trials      = []

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))

            # Skip nonsensical R:R (TP must be ≥ SL)
            if params.get("tp_pips", 1) < params.get("sl_pips", 0):
                continue

            config = BacktestConfig(
                symbol         = symbol,
                start_date     = start_date,
                end_date       = end_date,
                risk_pct       = params.get("risk_pct", 1.0),
                sl_pips        = params.get("sl_pips", 20.0),
                tp_pips        = params.get("tp_pips", 40.0),
            )
            bt = BacktestEngine(self.sm, config).run(notify_callback=_silent)
            s  = _score(bt, metric)

            trials.append({**params, metric: round(s, 4), "trades": bt.total_trades,
                           "max_dd_pct": bt.max_drawdown_pct})

            if s > best_score:
                best_score  = s
                best_params = dict(params)
                best_bt     = bt

            if i % 10 == 0:
                notify(f"[RiskOpt] {i}/{len(combos)} | best {metric}: {best_score:.4f}")

        notify(f"[RiskOpt] Done. Best {metric}: {best_score:.4f} | params: {best_params}")

        return OptimizationResult(
            optimizer   = "RiskParamOptimizer",
            symbol      = symbol,
            best_params = best_params,
            best_metric = round(best_score, 4),
            metric_name = metric,
            backtest    = best_bt,
            all_trials  = trials,
            elapsed_sec = round(time.perf_counter() - t0, 2),
        )

    def apply_to_profile(self, result: OptimizationResult) -> None:
        """Write the best risk params back into data/profile.json."""
        from manager.profile_manager import profile
        bp = result.best_params
        kwargs = {}
        if "risk_pct"   in bp: kwargs["risk_pct"]        = bp["risk_pct"]
        if "sl_pips"    in bp: kwargs["stop_loss_pips"]   = bp["sl_pips"]
        if "tp_pips"    in bp: kwargs["take_profit_pips"] = bp["tp_pips"]
        if kwargs:
            profile.update_risk(**kwargs)
            self.notify(f"[RiskOpt] Profile updated: {kwargs}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EnsembleWeightOptimizer
# ─────────────────────────────────────────────────────────────────────────────

# Regimes × strategies that make sense to tune
ENSEMBLE_REGIME_STRATEGIES = {
    "Strong Trend":                 ["Trend_Following", "Momentum", "Breakout", "Mean_Reversion", "Scalping"],
    "Ranging / Choppy":             ["Mean_Reversion", "Scalping", "Momentum", "Breakout", "Trend_Following"],
    "High Volatility Breakout":     ["Breakout", "Momentum", "Mean_Reversion", "Scalping"],
    "Low Volatility Consolidation": ["Mean_Reversion", "Scalping", "Momentum", "Breakout"],
}
WEIGHT_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]


class EnsembleWeightOptimizer:
    """
    Tunes the per-regime strategy weight multipliers used in MetaScorer's
    weighted-vote fallback path (REGIME_WEIGHTS dict).

    Approach: random search over weight assignments for each regime,
    evaluated by running a full backtest after patching REGIME_WEIGHTS.
    """

    def __init__(self, strategy_manager, notify: Callable = _silent):
        self.sm     = strategy_manager
        self.notify = notify

    def optimize(
        self,
        symbol:     str,
        regime:     str   = "Strong Trend",
        n_trials:   int   = 30,
        start_date: str   = "",
        end_date:   str   = "",
        metric:     str   = "sharpe",
        risk_pct:   float = 1.0,
        notify:     Callable = None,
    ) -> OptimizationResult:
        from strategies.models.meta_scorer import REGIME_WEIGHTS
        import math, copy

        notify = notify or self.notify
        strategies = ENSEMBLE_REGIME_STRATEGIES.get(
            regime, list(REGIME_WEIGHTS.get(regime, {}).keys())
        )
        if not strategies:
            return OptimizationResult(
                optimizer="EnsembleWeightOptimizer", symbol=symbol,
                best_params={}, best_metric=-math.inf, metric_name=metric,
                backtest=None, notes=f"Unknown regime: {regime}",
            )

        notify(
            f"[EnsembleOpt] Regime={regime} | "
            f"{len(strategies)} strategies | {n_trials} trials"
        )
        t0 = time.perf_counter()

        original_weights = copy.deepcopy(REGIME_WEIGHTS.get(regime, {}))
        best_score  = -math.inf
        best_params = {}
        best_bt     = None
        trials      = []

        import strategies.models.meta_scorer as ms_module

        for i in range(n_trials):
            weights = {s: random.choice(WEIGHT_VALUES) for s in strategies}
            ms_module.REGIME_WEIGHTS[regime] = weights

            bt = run_backtest(
                self.sm, symbol,
                start_date=start_date, end_date=end_date,
                risk_pct=risk_pct, notify_callback=_silent,
            )
            s = _score(bt, metric)
            trials.append({**weights, metric: round(s, 4)})

            if s > best_score:
                best_score  = s
                best_params = dict(weights)
                best_bt     = bt

            if (i + 1) % 10 == 0:
                notify(
                    f"[EnsembleOpt] {i+1}/{n_trials} | best {metric}: {best_score:.4f}"
                )

        # FIX 7: only apply new weights if at least one trial produced a finite
        # improvement.  When best_score is still -inf (no trades, bad data, etc.)
        # restore the original weights so we don't leave random noise in place.
        if best_params and best_score > -math.inf:
            ms_module.REGIME_WEIGHTS[regime] = best_params
            notify(f"[EnsembleOpt] Applied best weights for {regime}: {best_params}")
        else:
            ms_module.REGIME_WEIGHTS[regime] = original_weights
            notify(
                f"[EnsembleOpt] No valid trials for {regime} "
                f"(best score {best_score:.4f}) — original weights restored."
            )
            best_params = original_weights   # surface originals in the result

        return OptimizationResult(
            optimizer   = f"EnsembleWeightOptimizer[{regime}]",
            symbol      = symbol,
            best_params = best_params,
            best_metric = round(best_score, 4),
            metric_name = metric,
            backtest    = best_bt,
            all_trials  = trials,
            elapsed_sec = round(time.perf_counter() - t0, 2),
        )

    def optimize_all_regimes(
        self,
        symbol:     str,
        n_trials:   int   = 20,
        start_date: str   = "",
        end_date:   str   = "",
        metric:     str   = "sharpe",
        risk_pct:   float = 1.0,
        notify:     Callable = None,
    ) -> dict[str, OptimizationResult]:
        """Optimize all four regimes sequentially."""
        results = {}
        for regime in ENSEMBLE_REGIME_STRATEGIES:
            results[regime] = self.optimize(
                symbol, regime, n_trials, start_date, end_date, metric, risk_pct, notify,
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. WalkForwardOptimizer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """Single train/test window in a walk-forward run."""
    window_num:    int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    best_params:   dict
    train_score:   float
    test_score:    float
    test_backtest: Optional[BacktestResult]


@dataclass
class WalkForwardResult:
    """Aggregated result for a full walk-forward analysis."""
    symbol:       str
    param_space:  dict
    windows:      list[WalkForwardWindow]
    avg_test_score:  float
    std_test_score:  float
    best_oos_params: dict
    metric:       str
    elapsed_sec:  float

    def robustness_ratio(self) -> float:
        """
        Test score / Train score averaged over windows.
        Close to 1.0 → params generalise well.
        < 0.5       → likely overfit.
        """
        ratios = []
        for w in self.windows:
            if w.train_score > 0:
                ratios.append(w.test_score / w.train_score)
        return float(np.mean(ratios)) if ratios else 0.0

    def summary(self) -> str:
        lines = [
            f"{'═'*55}",
            f"Walk-Forward Analysis — {self.symbol}",
            f"Windows      : {len(self.windows)}",
            f"Metric       : {self.metric}",
            f"Avg OOS score: {self.avg_test_score:.4f}  ±{self.std_test_score:.4f}",
            f"Robustness   : {self.robustness_ratio():.2f}  (1.0 = perfect, <0.5 = overfit)",
            f"Elapsed      : {self.elapsed_sec:.1f}s",
            f"",
            f"Window results:",
        ]
        for w in self.windows:
            lines.append(
                f"  [{w.window_num}] train {w.train_start}→{w.train_end} "
                f"(IS={w.train_score:.3f}) | "
                f"test {w.test_start}→{w.test_end} "
                f"(OOS={w.test_score:.3f})"
            )
        lines.append(f"\nBest OOS params: {self.best_oos_params}")
        lines.append(f"{'═'*55}")
        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Validates any parameter set on rolling out-of-sample windows.

    The timeline is split into N overlapping [train | test] pairs.
    For each window: optimise on train data, evaluate on test data.
    Aggregates OOS (out-of-sample) scores to measure true robustness.

    Parameters
    ----------
    train_months : int   — length of each in-sample window in months
    test_months  : int   — length of each out-of-sample test window
    step_months  : int   — how far forward to slide each window
    """

    def __init__(
        self,
        strategy_manager,
        train_months: int = 6,
        test_months:  int = 2,
        step_months:  int = 2,
        notify: Callable  = _silent,
    ):
        self.sm           = strategy_manager
        self.train_months = train_months
        self.test_months  = test_months
        self.step_months  = step_months
        self.notify       = notify

    def run(
        self,
        symbol:      str,
        param_space: dict[str, list],
        data_start:  str,           # "YYYY-MM-DD"
        data_end:    str,           # "YYYY-MM-DD"
        n_trials:    int   = 20,
        metric:      str   = "sharpe",
        risk_pct:    float = 1.0,
        notify:      Callable = None,
    ) -> WalkForwardResult:
        notify = notify or self.notify
        t0     = time.perf_counter()
        windows_dates = self._build_windows(data_start, data_end)

        notify(
            f"[WFO] {symbol} | {len(windows_dates)} windows | "
            f"train={self.train_months}mo test={self.test_months}mo | {n_trials} trials/window"
        )

        strat_opt = StrategyParamOptimizer(self.sm, notify=_silent)
        risk_opt  = RiskParamOptimizer(self.sm, notify=_silent)

        windows   = []
        oos_scores = []

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows_dates, 1):
            notify(f"[WFO] Window {i}/{len(windows_dates)}: IS {tr_s}→{tr_e} | OOS {te_s}→{te_e}")

            # Determine which optimizer to use from the param_space keys
            is_risk = any(k in param_space for k in ("risk_pct", "sl_pips", "tp_pips"))

            if is_risk:
                is_res = risk_opt.optimize(
                    symbol, param_space, n_trials, tr_s, tr_e, metric, notify=_silent,
                )
            else:
                # Default: try StrategyParam for each active strategy
                strategy_name = list(self.sm.engines.keys())[0]   # first strategy
                is_res = strat_opt.optimize(
                    strategy_name, symbol, param_space, n_trials,
                    tr_s, tr_e, metric, risk_pct=risk_pct, notify=_silent,
                )

            # Evaluate best IS params on OOS window
            oos_config = BacktestConfig(
                symbol     = symbol,
                start_date = te_s,
                end_date   = te_e,
                risk_pct   = is_res.best_params.get("risk_pct", risk_pct),
                sl_pips    = is_res.best_params.get("sl_pips", 20.0),
                tp_pips    = is_res.best_params.get("tp_pips", 40.0),
            )
            oos_bt    = BacktestEngine(self.sm, oos_config).run(notify_callback=_silent)
            oos_score = _score(oos_bt, metric)
            oos_scores.append(oos_score)

            windows.append(WalkForwardWindow(
                window_num    = i,
                train_start   = tr_s,
                train_end     = tr_e,
                test_start    = te_s,
                test_end      = te_e,
                best_params   = is_res.best_params,
                train_score   = is_res.best_metric,
                test_score    = oos_score,
                test_backtest = oos_bt,
            ))

        # Best OOS params = window with highest OOS score
        best_window = max(windows, key=lambda w: w.test_score) if windows else None
        best_oos_params = best_window.best_params if best_window else {}

        return WalkForwardResult(
            symbol          = symbol,
            param_space     = param_space,
            windows         = windows,
            avg_test_score  = round(float(np.mean(oos_scores)), 4) if oos_scores else 0.0,
            std_test_score  = round(float(np.std(oos_scores)),  4) if oos_scores else 0.0,
            best_oos_params = best_oos_params,
            metric          = metric,
            elapsed_sec     = round(time.perf_counter() - t0, 2),
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_windows(self, data_start: str, data_end: str) -> list[tuple[str, str, str, str]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        from datetime import date
        import calendar

        def add_months(d: date, n: int) -> date:
            m = d.month + n
            y = d.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            last = calendar.monthrange(y, m)[1]
            return d.replace(year=y, month=m, day=min(d.day, last))

        try:
            start = datetime.strptime(data_start, "%Y-%m-%d").date()
            end   = datetime.strptime(data_end,   "%Y-%m-%d").date()
        except ValueError:
            return []

        windows = []
        cursor  = start

        while True:
            train_end  = add_months(cursor, self.train_months)
            test_end   = add_months(train_end, self.test_months)
            if test_end > end:
                break
            windows.append((
                cursor.isoformat(),
                train_end.isoformat(),
                train_end.isoformat(),
                test_end.isoformat(),
            ))
            cursor = add_months(cursor, self.step_months)

        return windows


# ─────────────────────────────────────────────────────────────────────────────
# 5. PortfolioOptimizer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioAllocation:
    """Capital allocation result for a portfolio of symbols."""
    symbols:             list[str]
    weights:             dict[str, float]   # symbol → fraction (sums to 1.0)
    expected_sharpe:     float
    expected_dd_pct:     float
    diversification:     float              # 1 − max_weight (higher = more diversified)
    per_symbol_results:  dict[str, BacktestResult]
    elapsed_sec:         float

    def summary(self) -> str:
        lines = [
            f"{'═'*55}",
            "Portfolio Allocation",
            f"Symbols       : {', '.join(self.symbols)}",
            f"Expected Sharpe: {self.expected_sharpe:.3f}",
            f"Expected DD    : {self.expected_dd_pct:.1f}%",
            f"Diversification: {self.diversification:.2f}",
            "",
            "Weights:",
        ]
        for sym, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {sym:<12} {w*100:5.1f}%")
        lines.append(f"{'═'*55}")
        return "\n".join(lines)


class PortfolioOptimizer:
    """
    Allocates capital across a list of symbols by maximising Sharpe or
    minimising max-drawdown using a Monte Carlo weight sweep.

    For each candidate weight vector:
    - Fetches pre-computed BacktestResult for each symbol (from cache or re-run)
    - Constructs a blended equity curve (weighted sum)
    - Scores the blended Sharpe / max-DD

    Constraints:
    - All weights ≥ min_weight
    - All weights ≤ max_weight
    - Sum of weights = 1.0
    """

    def __init__(
        self,
        strategy_manager,
        n_portfolios: int  = 500,
        min_weight:   float = 0.05,
        max_weight:   float = 0.60,
        notify: Callable    = _silent,
    ):
        self.sm           = strategy_manager
        self.n_portfolios = n_portfolios
        self.min_weight   = min_weight
        self.max_weight   = max_weight
        self.notify       = notify

    def optimize(
        self,
        symbols:    list[str],
        start_date: str   = "",
        end_date:   str   = "",
        risk_pct:   float = 1.0,
        objective:  str   = "sharpe",    # "sharpe" | "min_drawdown" | "calmar"
        notify:     Callable = None,
    ) -> PortfolioAllocation:
        notify = notify or self.notify
        t0     = time.perf_counter()

        if len(symbols) < 2:
            raise ValueError("PortfolioOptimizer needs at least 2 symbols.")

        notify(f"[PortfolioOpt] Backtesting {len(symbols)} symbols…")
        per_symbol: dict[str, BacktestResult] = {}
        equity_curves: dict[str, pd.Series]   = {}

        for sym in symbols:
            bt = run_backtest(
                self.sm, sym,
                start_date=start_date, end_date=end_date,
                risk_pct=risk_pct, notify_callback=_silent,
            )
            per_symbol[sym] = bt
            if bt and not bt.equity_curve.empty:
                equity_curves[sym] = bt.equity_curve
            notify(f"[PortfolioOpt]  {sym}: Sharpe={bt.sharpe_ratio:.2f} DD={bt.max_drawdown_pct:.1f}%")

        if not equity_curves:
            raise RuntimeError("No equity curve data available for any symbol.")

        # Align all curves on a common date index
        combined = pd.DataFrame(equity_curves).dropna(how="all").ffill().dropna()

        best_score  = -math.inf
        best_weights = {s: 1.0 / len(symbols) for s in symbols}

        notify(f"[PortfolioOpt] Monte Carlo sweep: {self.n_portfolios} portfolios…")

        for _ in range(self.n_portfolios):
            weights = self._random_weights(list(equity_curves.keys()))
            blended = sum(
                combined[sym] * w
                for sym, w in weights.items()
                if sym in combined.columns
            )

            score = self._score_blended(blended, objective)
            if score > best_score:
                best_score   = score
                best_weights = weights

        # Re-compute stats for the best portfolio
        blended_best = sum(
            combined[sym] * w
            for sym, w in best_weights.items()
            if sym in combined.columns
        )
        sharpe  = self._blended_sharpe(blended_best)
        dd_pct  = self._blended_max_dd_pct(blended_best)
        div     = 1.0 - max(best_weights.values())

        notify(f"[PortfolioOpt] Best portfolio Sharpe={sharpe:.3f} DD={dd_pct:.1f}% weights={best_weights}")

        return PortfolioAllocation(
            symbols            = symbols,
            weights            = best_weights,
            expected_sharpe    = round(sharpe,  3),
            expected_dd_pct    = round(dd_pct,  2),
            diversification    = round(div,     3),
            per_symbol_results = per_symbol,
            elapsed_sec        = round(time.perf_counter() - t0, 2),
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _random_weights(self, symbols: list[str]) -> dict[str, float]:
        """Sample a random weight vector that respects min/max constraints."""
        n = len(symbols)
        while True:
            raw = np.random.dirichlet(np.ones(n))
            raw = np.clip(raw, self.min_weight, self.max_weight)
            raw = raw / raw.sum()
            if (raw >= self.min_weight).all() and (raw <= self.max_weight).all():
                return dict(zip(symbols, raw.tolist()))

    def _score_blended(self, blended: pd.Series, objective: str) -> float:
        if objective == "sharpe":      return self._blended_sharpe(blended)
        if objective == "min_drawdown":return -self._blended_max_dd_pct(blended)
        if objective == "calmar":
            dd = self._blended_max_dd_pct(blended)
            ret = float(blended.iloc[-1] - blended.iloc[0]) if len(blended) > 1 else 0.0
            return (ret / dd) if dd > 0 else 0.0
        return self._blended_sharpe(blended)

    @staticmethod
    def _blended_sharpe(s: pd.Series, bars_per_year: int = 8760) -> float:
        rets = s.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0:
            return 0.0
        return float((rets.mean() / rets.std()) * math.sqrt(bars_per_year))

    @staticmethod
    def _blended_max_dd_pct(s: pd.Series) -> float:
        if s.empty:
            return 0.0
        roll_max = s.cummax()
        dd = (s - roll_max) / (roll_max + 1e-9)
        return float(abs(dd.min()) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Full optimization pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullOptimizationReport:
    """Aggregated report from run_full_optimization()."""
    symbol:           str
    strategy_results: dict[str, OptimizationResult]
    risk_result:      OptimizationResult
    ensemble_results: dict[str, OptimizationResult]
    wfo_result:       Optional[WalkForwardResult]
    elapsed_sec:      float

    def summary(self) -> str:
        lines = [
            f"{'═'*60}",
            f"FULL OPTIMIZATION REPORT — {self.symbol}",
            f"Total time : {self.elapsed_sec:.1f}s",
            f"{'═'*60}",
            "",
            "── Strategy Parameters ──────────────────────────────────────",
        ]
        for name, res in self.strategy_results.items():
            lines.append(f"  {name:<22} best Sharpe: {res.best_metric:.4f}")
        lines += [
            "",
            "── Risk Parameters ──────────────────────────────────────────",
            f"  best Calmar: {self.risk_result.best_metric:.4f}",
            f"  params     : {self.risk_result.best_params}",
            "",
            "── Ensemble Weights ─────────────────────────────────────────",
        ]
        for regime, res in self.ensemble_results.items():
            lines.append(f"  {regime:<30} best Sharpe: {res.best_metric:.4f}")

        if self.wfo_result:
            lines += ["", "── Walk-Forward Validation ──────────────────────────────"]
            lines.append(f"  Avg OOS Sharpe : {self.wfo_result.avg_test_score:.4f}")
            lines.append(f"  Robustness     : {self.wfo_result.robustness_ratio():.2f}")

        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    def save(self, path: str = None) -> str:
        out = path or f"data/full_optimization_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol":      self.symbol,
            "elapsed_sec": self.elapsed_sec,
            "risk_best":   self.risk_result.best_params,
            "strategy_best": {k: v.best_params for k, v in self.strategy_results.items()},
            "ensemble_best": {k: v.best_params for k, v in self.ensemble_results.items()},
        }
        if self.wfo_result:
            payload["wfo_avg_oos"] = self.wfo_result.avg_test_score
            payload["wfo_robustness"] = self.wfo_result.robustness_ratio()
        Path(out).write_text(json.dumps(payload, indent=2, default=str))
        return out


def run_full_optimization(
    strategy_manager,
    symbol:          str,
    start_date:      str   = "",
    end_date:        str   = "",
    run_wfo:         bool  = False,    # WFO is slow; opt-in
    strategy_names:  list  = None,
    n_trials:        int   = 20,
    notify:          Callable = print,
) -> FullOptimizationReport:
    """
    One-call convenience wrapper that runs all five optimizers sequentially
    and returns a FullOptimizationReport.

    Usage (from ARIA chat):
        from optimizer import run_full_optimization
        report = run_full_optimization(strategy_manager, "EURUSD",
                                       start_date="2024-01-01",
                                       end_date="2024-12-31")
        print(report.summary())
        report.save()

    Parameters
    ----------
    run_wfo : bool
        Walk-forward validation is disabled by default (slow).
        Set True to enable.
    strategy_names : list[str]
        Which strategies to optimise. Defaults to all defined in
        STRATEGY_PARAM_SPACES.
    """
    t0 = time.perf_counter()
    strats = strategy_names or list(STRATEGY_PARAM_SPACES.keys())

    notify(f"\n{'='*60}")
    notify(f"ARIA Full Optimization — {symbol}")
    notify(f"Period: {start_date or 'all'} → {end_date or 'all'}")
    notify(f"{'='*60}\n")

    # 1. Strategy params
    strategy_results = {}
    strat_opt = StrategyParamOptimizer(strategy_manager, notify=notify)
    for sname in strats:
        if sname in strategy_manager.engines:
            notify(f"\n[1/5] Optimizing {sname} params…")
            strategy_results[sname] = strat_opt.optimize(
                sname, symbol,
                start_date=start_date, end_date=end_date,
                max_trials=n_trials, notify=notify,
            )

    # 2. Risk params
    notify("\n[2/5] Optimizing risk params…")
    risk_opt    = RiskParamOptimizer(strategy_manager, notify=notify)
    risk_result = risk_opt.optimize(
        symbol, n_trials=n_trials,
        start_date=start_date, end_date=end_date,
        metric="calmar", notify=notify,
    )
    risk_opt.apply_to_profile(risk_result)

    # 3. Ensemble weights
    notify("\n[3/5] Optimizing ensemble weights…")
    ens_opt      = EnsembleWeightOptimizer(strategy_manager, notify=notify)
    ens_results  = ens_opt.optimize_all_regimes(
        symbol, n_trials=n_trials,
        start_date=start_date, end_date=end_date,
        notify=notify,
    )

    # 4. Walk-forward validation (optional)
    wfo_result = None
    if run_wfo and start_date and end_date:
        notify("\n[4/5] Walk-forward validation…")
        wfo = WalkForwardOptimizer(strategy_manager, notify=notify)
        wfo_result = wfo.run(
            symbol, param_space=RISK_PARAM_SPACE,
            data_start=start_date, data_end=end_date,
            n_trials=n_trials, metric="sharpe",
        )
    else:
        notify("\n[4/5] Walk-forward skipped (set run_wfo=True to enable).")

    notify("\n[5/5] Compiling report…")
    report = FullOptimizationReport(
        symbol           = symbol,
        strategy_results = strategy_results,
        risk_result      = risk_result,
        ensemble_results = ens_results,
        wfo_result       = wfo_result,
        elapsed_sec      = round(time.perf_counter() - t0, 2),
    )
    notify("\n" + report.summary())
    return report