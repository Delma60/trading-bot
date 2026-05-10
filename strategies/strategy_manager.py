"""
strategies/strategy_manager.py

Key change from previous version
---------------------------------
A SymbolRegistry is created from the connected broker on init and:
  1. Injected into ArbitrageStrategy so it uses broker-sourced symbols
     instead of a hardcoded universe list.
  2. Exposed as self.symbol_registry for any other component that needs
     a live symbol catalogue (PortfolioManager, ProactiveEngine, etc.).

Everything else is unchanged.
"""

import pandas as pd

from inspect import signature
from strategies.arbitrage import ArbitrageStrategy
from strategies.breakout import BreakoutStrategy
from strategies.momentum import MomentumStrategy
from strategies.scalping import ScalpingStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.news_trading import NewsTradingStrategy
from trader import Trader
from .mean_reversion import MeanReversionStrategy
from strategies.features.feature_engineer import FeatureEngineer
from strategies.models.lstm_predictor import LSTMPredictor
from strategies.models.meta_scorer import MetaScorer
from manager.symbol_registry import SymbolRegistry          # ← NEW
import threading
import time
from manager.profile_manager import profile as _profile


class DummyStrategy:
    """Safe placeholder for strategies not yet implemented."""
    def analyze(self, df: pd.DataFrame) -> dict:
        return {"action": "WAIT", "confidence": 0.0,
                "reason": "Strategy not yet implemented."}


class OHLCVCache:
    """
    Thread-safe time-to-live cache for OHLCV DataFrames.
    Prevents redundant MT5 calls when multiple strategies analyse the
    same symbol in the same scan cycle.
    """

    def __init__(self, ttl_seconds: int = 60):
        self._store: dict[tuple, tuple[float, object]] = {}
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()

    def fetch(self, broker, symbol: str, timeframe: str, num_bars: int = 1000):
        key = (symbol.upper(), timeframe)
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.monotonic() - entry[0]) < self._ttl:
                return entry[1]

        df = broker.ohclv_data(symbol, timeframe=timeframe, num_bars=num_bars)

        if df is not None and not df.empty:
            with self._lock:
                self._store[key] = (time.monotonic(), df)

        return df

    def invalidate(self, symbol: str = None, timeframe: str = None):
        with self._lock:
            if symbol is None:
                self._store.clear()
            else:
                key = (symbol.upper(), timeframe or "H1")
                self._store.pop(key, None)


class MTFConfluenceEngine:
    """Multi-Timeframe Confluence engine (Feature #4)."""
    TIMEFRAMES = ["M15", "H1", "H4", "D1"]

    def __init__(self, broker):
        self.broker = broker
        self.broker_timeout_seconds = 5.0

    def get_confluence_score(self, symbol: str, cache=None) -> dict:
        signals = {}
        for tf in self.TIMEFRAMES:
            try:
                if cache:
                    df = cache.get_raw_ohlcv(symbol, tf)
                else:
                    from threading import Thread
                    result_container = [None]
                    exception_container = [None]

                    def fetch_with_timeout():
                        try:
                            result_container[0] = self.broker.get_historical_rates(
                                symbol, tf, 50
                            )
                        except Exception as e:
                            exception_container[0] = e

                    thread = Thread(target=fetch_with_timeout, daemon=True)
                    thread.start()
                    thread.join(timeout=self.broker_timeout_seconds)

                    if thread.is_alive():
                        signals[tf] = "NEUTRAL"
                        continue
                    if exception_container[0]:
                        signals[tf] = "NEUTRAL"
                        continue
                    df = result_container[0]

                if df is None or df.empty or len(df) < 25:
                    signals[tf] = "NEUTRAL"
                    continue

                closes = df["close"].values
                sma20 = sum(closes[-20:]) / 20
                sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma20
                current = closes[-1]

                if current > sma20 and sma20 > sma50:
                    signals[tf] = "BUY"
                elif current < sma20 and sma20 < sma50:
                    signals[tf] = "SELL"
                else:
                    signals[tf] = "NEUTRAL"
            except Exception:
                signals[tf] = "NEUTRAL"

        directions = list(signals.values())
        dominant = max(set(directions), key=directions.count)
        active_tfs = [d for d in directions if d != "NEUTRAL"]
        alignment = (
            active_tfs.count(dominant) / len(self.TIMEFRAMES) if active_tfs else 0.0
        )

        return {
            "direction": dominant,
            "alignment": alignment,
            "signals":   signals,
            "tradeable": alignment >= _profile.scanner().mtf_min_alignment,
        }


class StrategyManager:
    """
    Orchestrates all trading strategies, the LSTM deep-learning predictor,
    and the XGBoost meta-scorer into a single, calibrated trading signal.

    New in this version
    -------------------
    self.symbol_registry : SymbolRegistry
        Built from the live broker on init.  Replaces every hardcoded
        symbol list in the codebase.  Injected into ArbitrageStrategy
        so it searches the broker's real catalogue for correlated pairs.
    """

    strategies = [
        "Mean_Reversion",
        "Momentum",
        "Breakout",
        "Scalping",
        "News_Trading",
        "Sentiment_Analysis",
        "Arbitrage",
    ]

    descriptions = {
        "Mean_Reversion":     "Mean Reversion: Buys oversold dips, sells overbought rips.",
        "Momentum":           "Momentum: Buys when price trends up, sells when trending down.",
        "Breakout":           "Breakout: Buys above resistance, sells below support.",
        "Scalping":           "Scalping: Small profits from frequent short-term trades.",
        "News_Trading":       "News Trading: Trades based on economic news (stub).",
        "Sentiment_Analysis": "Sentiment Analysis: Trades based on market sentiment (stub).",
        "Arbitrage":          "Arbitrage: Exploits price differences between markets.",
    }

    def __init__(self, broker: Trader, cache=None, notify_callback=print):
        self.broker  = broker
        self.cache   = cache
        self.notify  = notify_callback

        self._model_lock = threading.Lock()

        # ── Symbol Registry ──────────────────────────────────────────────────
        # Single, broker-aware catalogue shared across all components.
        # ArbitrageStrategy receives a reference so it never needs a hardcoded
        # universe list.
        self.symbol_registry = SymbolRegistry(broker)

        # ML/DL components
        self.lstm     = LSTMPredictor()
        self.meta     = MetaScorer()
        self.features = FeatureEngineer()
        self._ohlcv_cache = None if cache is not None else OHLCVCache(ttl_seconds=60)

        # Unsupervised learning engine
        try:
            from manager.unsupervised_learner import UnsupervisedLearner
            self.learner = UnsupervisedLearner()
        except ImportError:
            self.learner = None

        # ── Strategy engines ─────────────────────────────────────────────────
        # ArbitrageStrategy receives the registry — no more hardcoded universe.
        self.engines: dict[str, object] = {
            "Mean_Reversion":     MeanReversionStrategy(),
            "Momentum":           MomentumStrategy(),
            "Breakout":           BreakoutStrategy(),
            "Scalping":           ScalpingStrategy(),
            "News_Trading":       DummyStrategy(),
            "Sentiment_Analysis": DummyStrategy(),
            "Arbitrage":          ArbitrageStrategy(
                                      symbol_registry=self.symbol_registry
                                  ),
            "Trend_Following":    TrendFollowingStrategy(),
        }

        self.active_ensemble_strategies = [
            k for k in self.engines
            if k not in ("News_Trading", "Sentiment_Analysis")
        ]

        from manager.risk_manager import MarketConditionFilter
        self._mc_filter  = MarketConditionFilter(broker)
        self._mtf_engine = MTFConfluenceEngine(broker)

    # ------------------------------------------------------------------
    # Public API  (identical to previous version)
    # ------------------------------------------------------------------

    def get_strategy_description(self, strategy_name: str) -> str:
        return self.descriptions.get(strategy_name, "Unknown strategy.")

    def execute_strategy(self, strategy_name: str):
        return self.engines.get(strategy_name, DummyStrategy())

    def record_trade_outcome(self, feature_vector, action: str, profit: float):
        label = action if profit > 0 else "WAIT"
        self.meta.collect_sample(feature_vector, label)

        sample_count = len(self.meta._samples)
        if sample_count >= 200 and sample_count % 50 == 0:
            acc = self.meta.train(force=True)
            if acc > 0:
                self.notify(
                    f"[MetaScorer] Retrained on {sample_count} samples. "
                    f"Validation accuracy: {acc:.1%}",
                    priority="normal",
                )

    def continuous_learning_routine(self, symbol: str):
        raw_df = self._ohlcv_cache.fetch(self.broker, symbol, "H1", num_bars=500)
        if raw_df is None or raw_df.empty:
            self.notify(
                f"[StrategyManager] Could not fetch OHLCV for {symbol} to retrain."
            )
            return

        feat_df = FeatureEngineer.compute(raw_df)
        if feat_df.empty:
            self.notify(
                f"[StrategyManager] Feature engineering failed for {symbol}."
            )
            return

        with self._model_lock:
            self.lstm.train(feat_df, symbol=symbol, force=True)

        sample_count = len(self.meta._samples)
        if sample_count < 200:
            self.notify(
                f"[StrategyManager] Skipping meta-scorer retrain for {symbol}: "
                f"only {sample_count} samples available."
            )
            return

        with self._model_lock:
            acc = self.meta.train(force=True)
        if acc > 0:
            self.notify(
                f"[StrategyManager] Retrained models for {symbol}. "
                f"Meta accuracy: {acc:.1%}"
            )

    def check_signals(
        self,
        symbol:        str,
        strategy:      str  = "Mean_Reversion",
        timeframe:     str  = "H1",
        use_ensemble:  bool = True,
        retrain_lstm:  bool = False,
    ) -> dict:
        # 1. Fetch OHLCV / features
        if self.cache is not None:
            feat_df = self.cache.get_features(symbol)
            raw_df  = self.cache.get_raw_ohlcv(symbol)
            if feat_df is None or feat_df.empty:
                return {
                    "action":     "WAIT",
                    "confidence": 0.0,
                    "reason":     f"Features for {symbol} are still warming up.",
                }
        else:
            raw_df = self._ohlcv_cache.fetch(
                self.broker, symbol, timeframe, num_bars=1000
            )
            if raw_df is None or raw_df.empty:
                return {
                    "action":     "WAIT",
                    "confidence": 0.0,
                    "reason":     f"Could not fetch OHLCV data for {symbol}.",
                }
            feat_df = FeatureEngineer.compute(raw_df)
            if feat_df.empty:
                return {
                    "action":     "WAIT",
                    "confidence": 0.0,
                    "reason":     "Feature engineering produced an empty DataFrame.",
                }

        # Market condition pre-filter
        is_suitable, mc_reason = self._mc_filter.is_market_suitable(symbol)
        if not is_suitable:
            return {"action": "WAIT", "confidence": 0.0, "reason": mc_reason}

        # MTF confluence check
        mtf_data = self._mtf_engine.get_confluence_score(symbol)
        if not mtf_data["tradeable"]:
            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason": (
                    f"MTF Confluence too low ({mtf_data['alignment']:.0%}). "
                    f"Macro trend is unclear."
                ),
            }

        # Single-strategy fallback
        if not use_ensemble:
            if strategy not in self.engines:
                self.notify(
                    f"[StrategyManager] Unknown strategy '{strategy}'. "
                    f"Defaulting to Mean_Reversion."
                )
                strategy = "Mean_Reversion"
            return self.engines[strategy].analyze(raw_df)

        # Feed features to unsupervised learner
        if self.learner:
            latest_features = feat_df.iloc[-1]
            self.learner.ingest_market_bar(latest_features)

        # Collect signals from every strategy engine
        strategy_signals: dict[str, dict] = {}
        for name in self.active_ensemble_strategies:
            engine = self.engines[name]
            try:
                method_sig = signature(engine.analyze)
                args = [feat_df]
                if len(method_sig.parameters) >= 2:
                    args.append(symbol)
                if len(method_sig.parameters) >= 3:
                    args.append(self.broker)
                strategy_signals[name] = engine.analyze(*args)
            except Exception as exc:
                self.notify(f"[StrategyManager] {name} raised: {exc}")
                strategy_signals[name] = {"action": "WAIT", "confidence": 0.0}

        # Train / load LSTM
        with self._model_lock:
            self.lstm.train(feat_df, symbol=symbol, force=retrain_lstm)
            lstm_pred = self.lstm.predict(feat_df, symbol=symbol)

        # Meta-scorer final decision
        market_snapshot = feat_df.iloc[-1]
        regime = "Unknown"
        if self.learner:
            regime = self.learner.get_current_regime()

        with self._model_lock:
            final = self.meta.score(
                strategy_signals, lstm_pred, market_snapshot, regime=regime
            )

        fv = self.meta.build_feature_vector(
            strategy_signals, lstm_pred, market_snapshot
        )

        vote_summary = ", ".join(
            f"{k}={v.get('action','?')}({v.get('confidence', 0):.0%})"
            for k, v in strategy_signals.items()
        )
        reason = (
            f"[{final['source']}] {final['action']} @ "
            f"{final['confidence']:.1%} confidence. "
            f"LSTM: {lstm_pred['direction']} ({lstm_pred['confidence']:.1%}). "
            f"Strategies: {vote_summary}."
        )

        return {
            "action":           final["action"],
            "confidence":       final["confidence"],
            "source":           final["source"],
            "reason":           reason,
            "lstm_prediction":  lstm_pred,
            "strategy_signals": strategy_signals,
            "feature_vector":   fv,
        }