import pandas as pd

from strategies.arbitrage import ArbitrageStrategy
from strategies.breakout import BreakoutStrategy
from strategies.momentum import MomentumStrategy
from strategies.scalping import ScalpingStrategy
from strategies.trend_following import TrendFollowingStrategy
from trader import Trader
from .mean_reversion import MeanReversionStrategy
from strategies.features.feature_engineer import FeatureEngineer
from strategies.models.lstm_predictor import LSTMPredictor
from strategies.models.meta_scorer import MetaScorer


class DummyStrategy:
    """Safe placeholder for strategies not yet implemented."""
    def analyze(self, df: pd.DataFrame) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": "Strategy not yet implemented."}


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

    def __init__(self, broker: Trader, notify_callback=print):
        self.broker  = broker
        self.notify  = notify_callback

        # ML/DL components — instantiated once and reused across calls
        self.lstm = LSTMPredictor()
        self.meta = MetaScorer()

        # Strategy engines — instantiated once to avoid per-call overhead
        self.engines: dict[str, object] = {
            "Mean_Reversion":     MeanReversionStrategy(),
            "Momentum":           MomentumStrategy(),
            "Breakout":           BreakoutStrategy(),
            "Scalping":           ScalpingStrategy(),
            "News_Trading":       DummyStrategy(),
            "Sentiment_Analysis": DummyStrategy(),
            "Arbitrage":          ArbitrageStrategy(),
            "Trend_Following":    TrendFollowingStrategy(),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_strategy_description(self, strategy_name: str) -> str:
        return self.descriptions.get(strategy_name, "Unknown strategy.")

    def execute_strategy(self, strategy_name: str):
        """Return the pre-loaded strategy engine object."""
        return self.engines.get(strategy_name, DummyStrategy())

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
        # 1. Fetch raw OHLCV ──────────────────────────────────────────────────
        raw_df = self.broker.ohclv_data(symbol, timeframe=timeframe, num_bars=1000)
        if raw_df is None or raw_df.empty:
            return {
                "action": "WAIT",
                "confidence": 0.0,
                "reason": f"Could not fetch OHLCV data for {symbol}.",
            }

        # 2. Single-strategy fallback ─────────────────────────────────────────
        if not use_ensemble:
            if strategy not in self.engines:
                self.notify(f"[StrategyManager] ⚠️ Unknown strategy '{strategy}'. Defaulting to Mean_Reversion.")
                strategy = "Mean_Reversion"
            return self.engines[strategy].analyze(raw_df)

        # 3. Engineer features ────────────────────────────────────────────────
        feat_df = FeatureEngineer.compute(raw_df)
        if feat_df.empty:
            return {
                "action": "WAIT",
                "confidence": 0.0,
                "reason": "Feature engineering produced an empty DataFrame.",
            }

        # 4. Collect signals from every strategy engine ───────────────────────
        strategy_signals: dict[str, dict] = {}
        for name, engine in self.engines.items():
            try:
                strategy_signals[name] = engine.analyze(feat_df)
            except Exception as exc:
                self.notify(f"[StrategyManager] ⚠️ {name} raised: {exc}")
                strategy_signals[name] = {"action": "WAIT", "confidence": 0.0}

        # 5. Train LSTM (loads cached weights on subsequent calls) ─────────────
        self.lstm.train(feat_df, symbol=symbol, force=retrain_lstm)
        lstm_pred = self.lstm.predict(feat_df, symbol=symbol)

        # 6. Meta-scorer makes the final call ─────────────────────────────────
        market_snapshot = feat_df.iloc[-1]
        final           = self.meta.score(strategy_signals, lstm_pred, market_snapshot)

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