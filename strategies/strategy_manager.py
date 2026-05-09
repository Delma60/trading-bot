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
import threading
import time

class DummyStrategy:
    """Safe placeholder for strategies not yet implemented."""
    def analyze(self, df: pd.DataFrame) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": "Strategy not yet implemented."}


class OHLCVCache:
    """
    Thread-safe time-to-live cache for OHLCV DataFrames.
 
    Prevents redundant MT5 calls when multiple strategies analyse the
    same symbol in the same scan cycle.
 
    Usage in StrategyManager.__init__:
        self._ohlcv_cache = OHLCVCache(ttl_seconds=60)
 
    Usage in check_signals() — replace the broker.ohclv_data() call:
        raw_df = self._ohlcv_cache.fetch(
            self.broker, symbol, timeframe, num_bars=1000
        )
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
                return entry[1]   # cache hit
 
        # Cache miss — fetch from broker (outside the lock so we don't block)
        df = broker.ohclv_data(symbol, timeframe=timeframe, num_bars=num_bars)
 
        if df is not None and not df.empty:
            with self._lock:
                self._store[key] = (time.monotonic(), df)
 
        return df
 
    def invalidate(self, symbol: str = None, timeframe: str = None):
        """Force a refresh on next fetch. Call after known market events."""
        with self._lock:
            if symbol is None:
                self._store.clear()
            else:
                key = (symbol.upper(), timeframe or "H1")
                self._store.pop(key, None)
 

class MTFConfluenceEngine:
    """Multi-Timeframe Confluence engine — ensures alignment across HTF before trades (Feature #4)."""
    TIMEFRAMES = ["M15", "H1", "H4", "D1"]
    
    def __init__(self, broker):
        self.broker = broker
        self.broker_timeout_seconds = 5.0  # Per-call timeout for broker

    def get_confluence_score(self, symbol: str, cache=None) -> dict:
        """Analyze alignment across timeframes and return a tradeable verdict."""
        signals = {}
        for tf in self.TIMEFRAMES:
            try:
                if cache:
                    df = cache.get_raw_ohlcv(symbol)  # free, no MT5 call
                else:
                    # Add timeout to prevent hanging broker calls
                    from threading import Thread
                    result_container = [None]
                    exception_container = [None]
                    
                    def fetch_with_timeout():
                        try:
                            result_container[0] = self.broker.get_historical_rates(symbol, tf, 50)
                        except Exception as e:
                            exception_container[0] = e
                    
                    thread = Thread(target=fetch_with_timeout, daemon=True)
                    thread.start()
                    thread.join(timeout=self.broker_timeout_seconds)
                    
                    if thread.is_alive():
                        # Thread still running = broker hung up
                        signals[tf] = "NEUTRAL"
                        continue
                    
                    if exception_container[0]:
                        signals[tf] = "NEUTRAL"
                        continue
                    
                    df = result_container[0]
                if df is None or df.empty or len(df) < 25:
                    signals[tf] = "NEUTRAL"
                    continue
                    
                closes = df['close'].values
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
        
        # Calculate alignment ignoring NEUTRALs
        active_tfs = [d for d in directions if d != "NEUTRAL"]
        alignment = active_tfs.count(dominant) / len(self.TIMEFRAMES) if active_tfs else 0.0
        
        return {
            "direction": dominant,
            "alignment": alignment,
            "signals": signals,
            "tradeable": alignment >= 0.50  # 2 out of 4 must align (reduced from 3/4)
        }


class StrategyManager:
    """
    Orchestrates all trading strategies, the LSTM deep-learning predictor,
    and the XGBoost meta-scorer into a single, calibrated trading signal.

    Two modes
    ---------
    use_ensemble=True  (default)
        1. Fetch raw OHLCV and engineer ~40 features.
        2. Run every strategy engine and collect their signals.
        3. Run the LSTM for a next-bar direction prediction.
        4. Feed all signals + market context into the MetaScorer.
        5. Return a unified {"action", "confidence", "reason", ...} dict.

    use_ensemble=False
        Falls back to the original single-strategy flow.

    Meta-model training
    -------------------
    After enough live/paper bars have been labelled you can train the meta-model:

        manager.meta.collect_sample(feature_vector, label)   # call each bar
        manager.meta.train()                                  # when ready
    """

    # Strategies exposed to the Portfolio Manager
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

        # ML/DL components — instantiated once and reused across calls
        self.lstm = LSTMPredictor()
        self.meta = MetaScorer()
        self.features = FeatureEngineer()
        self._ohlcv_cache = None if cache is not None else OHLCVCache(ttl_seconds=60)

        # Unsupervised learning engine
        try:
            from manager.unsupervised_learner import UnsupervisedLearner
            self.learner = UnsupervisedLearner()
        except ImportError:
            self.learner = None

        # Strategy engines — instantiated once to avoid per-call overhead
        self.engines: dict[str, object] = {
            "Mean_Reversion":     MeanReversionStrategy(),
            "Momentum":           MomentumStrategy(),
            "Breakout":           BreakoutStrategy(),
            "Scalping":           ScalpingStrategy(),
            "News_Trading":       DummyStrategy(),  # Exclude unimplemented news strategy
            "Sentiment_Analysis": DummyStrategy(),
            "Arbitrage":          ArbitrageStrategy(),
            "Trend_Following":    TrendFollowingStrategy(),
        }
        
        # Exclude unimplemented strategies from ensemble voting
        self.active_ensemble_strategies = [k for k in self.engines if k not in ("News_Trading", "Sentiment_Analysis")]

        # Market filters — instantiated once to avoid fresh objects on every call
        from manager.risk_manager import MarketConditionFilter
        self._mc_filter = MarketConditionFilter(broker)
        self._mtf_engine = MTFConfluenceEngine(broker)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_strategy_description(self, strategy_name: str) -> str:
        return self.descriptions.get(strategy_name, "Unknown strategy.")

    def execute_strategy(self, strategy_name: str):
        """Return the pre-loaded strategy engine object."""
        return self.engines.get(strategy_name, DummyStrategy())

    def record_trade_outcome(self, feature_vector, action: str, profit: float):
        """
        Call this when a position CLOSES.

        Parameters
        ----------
        feature_vector : np.ndarray
            The vector returned in check_signals()["feature_vector"] at the
            time the trade was opened.  Store it in a dict keyed by ticket
            when the trade closes.
        action : str
            "BUY" or "SELL" — the direction that was taken.
        profit : float
            Final realised P&L of the closed position.

        The label passed to MetaScorer is:
            - the actual direction ("BUY"/"SELL") if profitable
            - "WAIT" if the trade was a loss (meaning the model should have
            waited, regardless of direction)
        """
        label = action if profit > 0 else "WAIT"
        self.meta.collect_sample(feature_vector, label)

        # Auto-train once we cross the minimum sample threshold
        sample_count = len(self.meta._samples)
        if sample_count >= 200 and sample_count % 50 == 0:
            # Every 50 new samples after the 200-sample threshold, retrain
            acc = self.meta.train(force=True)
            if acc > 0:
                self.notify(
                    f"[MetaScorer] Retrained on {sample_count} samples. "
                    f"Validation accuracy: {acc:.1%}",
                    priority="normal",
                )

    def continuous_learning_routine(self, symbol: str):
        """Retrain strategy models and meta-scorer for a symbol using recent history."""
        raw_df = self._ohlcv_cache.fetch(self.broker, symbol, "H1", num_bars=500)
        if raw_df is None or raw_df.empty:
            self.notify(f"[StrategyManager] Could not fetch OHLCV for {symbol} to retrain.")
            return

        feat_df = FeatureEngineer.compute(raw_df)
        if feat_df.empty:
            self.notify(f"[StrategyManager] Feature engineering failed for {symbol} during retraining.")
            return

        self.lstm.train(feat_df, symbol=symbol, force=True)

        sample_count = len(self.meta._samples)
        if sample_count < 200:
            self.notify(
                f"[StrategyManager] Skipping meta-scorer retrain for {symbol}: only {sample_count} samples available."
            )
            return

        acc = self.meta.train(force=True)
        if acc > 0:
            self.notify(f"[StrategyManager] Retrained models for {symbol}. Meta accuracy: {acc:.1%}")

    def check_signals(
        self,
        symbol:        str,
        strategy:      str  = "Mean_Reversion",
        timeframe:     str  = "H1",
        use_ensemble:  bool = True,
        retrain_lstm:  bool = False,
    ) -> dict:
        """
        Fetch the latest data and return a trading decision.

        Parameters
        ----------
        symbol        : Instrument ticker (e.g. "EURUSD", "BTCUSDT").
        strategy      : Strategy to use in single-strategy mode.
        timeframe     : Broker timeframe string (default "H1").
        use_ensemble  : When True, runs all strategies + LSTM + meta-scorer.
        retrain_lstm  : When True, forces the LSTM to retrain even if weights exist.

        Returns
        -------
        {
            "action":           "BUY" | "SELL" | "WAIT",
            "confidence":       float,          # 0.0–1.0
            "source":           str,            # "meta_model" | "weighted_vote" | strategy name
            "reason":           str,
            "lstm_prediction":  dict,           # only in ensemble mode
            "strategy_signals": dict,           # only in ensemble mode
        }
        """
        # 1. Fetch raw OHLCV / pre-computed features from cache if available
        if self.cache is not None:
            feat_df = self.cache.get_features(symbol)
            raw_df = self.cache.get_raw_ohlcv(symbol)
            if feat_df is None or feat_df.empty:
                return {
                    "action": "WAIT",
                    "confidence": 0.0,
                    "reason": f"Features for {symbol} are still warming up in cache.",
                }
        else:
            raw_df = self._ohlcv_cache.fetch(self.broker, symbol, timeframe, num_bars=1000)
            if raw_df is None or raw_df.empty:
                return {
                    "action": "WAIT",
                    "confidence": 0.0,
                    "reason": f"Could not fetch OHLCV data for {symbol}.",
                }
            feat_df = FeatureEngineer.compute(raw_df)
            if feat_df.empty:
                return {
                    "action": "WAIT",
                    "confidence": 0.0,
                    "reason": "Feature engineering produced an empty DataFrame.",
                }

        # (NEW) Market Condition Pre-Filter (Feature #2)
        is_suitable, mc_reason = self._mc_filter.is_market_suitable(symbol)
        if not is_suitable:
            return {"action": "WAIT", "confidence": 0.0, "reason": mc_reason}

        # (NEW) Multi-Timeframe Confluence Check (Feature #4)
        mtf_data = self._mtf_engine.get_confluence_score(symbol)
        
        # Force WAIT if macro alignment is poor
        if not mtf_data["tradeable"]:
             return {"action": "WAIT", "confidence": 0.0, "reason": f"MTF Confluence too low ({mtf_data['alignment']:.0%}). Macro trend is unclear."}

        # 2. Single-strategy fallback ─────────────────────────────────────────
        if not use_ensemble:
            if strategy not in self.engines:
                self.notify(f"[StrategyManager] ⚠️ Unknown strategy '{strategy}'. Defaulting to Mean_Reversion.")
                strategy = "Mean_Reversion"
            return self.engines[strategy].analyze(raw_df)

        # 3.5. Feed features to unsupervised learner ──────────────────────────
        if self.learner:
            latest_features = feat_df.iloc[-1]  # Most recent bar's features
            regime = self.learner.ingest_market_bar(latest_features)
            # Regime will be used by reasoning engine for context-aware decisions

        # 4. Collect signals from every strategy engine ───────────────────────
        strategy_signals: dict[str, dict] = {}
        for name in self.active_ensemble_strategies:  # Use only implemented strategies
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
                self.notify(f"[StrategyManager] ⚠️ {name} raised: {exc}")
                strategy_signals[name] = {"action": "WAIT", "confidence": 0.0}

        # 5. Train LSTM (loads cached weights on subsequent calls) ─────────────
        self.lstm.train(feat_df, symbol=symbol, force=retrain_lstm)
        lstm_pred = self.lstm.predict(feat_df, symbol=symbol)

        # 6. Meta-scorer makes the final call ─────────────────────────────────
        market_snapshot = feat_df.iloc[-1]
        regime = "Unknown"
        if self.learner:
            regime = self.learner.get_current_regime()
            
        final = self.meta.score(strategy_signals, lstm_pred, market_snapshot, regime=regime)

        # 7. Optionally accumulate a training sample for the meta-model ────────
        fv = self.meta.build_feature_vector(strategy_signals, lstm_pred, market_snapshot)
        # Label will be added externally once the trade outcome is known:
        #   manager.meta.collect_sample(fv, "BUY")  →  manager.meta.train()

        # 8. Build a rich, readable response ──────────────────────────────────
        vote_summary = ", ".join(
            f"{k}={v.get('action','?')}({v.get('confidence', 0):.0%})"
            for k, v in strategy_signals.items()
        )
        reason = (
            f"[{final['source']}] {final['action']} @ {final['confidence']:.1%} confidence. "
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
            "feature_vector":   fv,   # keep for meta-model sample collection
        }