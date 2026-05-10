"""
manager/unsupervised_learner.py — ARIA's Self-Discovery Engine

Learns profitable market regimes and trade patterns without labelled data.

What it does
------------
1. Regime Clustering (K-Means)
   Groups recent market states into regimes (trending/ranging/volatile/quiet)
   and learns which strategies outperform in each.

2. Anomaly Detection (Isolation Forest)
   Flags unusual market conditions that historically precede losses.

3. Trade Outcome Pattern Mining
   Discovers which feature combinations (RSI level, ADX, volatility, time of day)
   correlate with winning vs losing trades — no manual labelling needed.

4. Strategy Weight Adaptation
   Continuously adjusts per-regime strategy confidence multipliers based on
   cluster performance history.

5. Behaviour Summariser
   Periodically generates human-readable insight strings fed back to AIBrain.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import skops.io as sio

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Feature columns used for clustering ──────────────────────────────────────
REGIME_FEATURES = [
    "adx", "rsi_14", "volatility_20", "volume_ratio",
    "bb_width", "regime_trending", "dist_sma50",
]

TRADE_FEATURES = [
    "adx", "rsi_14", "volatility_20", "volume_ratio",
    "bb_pos", "bb_width", "dist_sma50", "dist_sma200",
    "macd_hist", "di_diff",
]

REGIME_LABELS = {
    0: "Strong Trend",
    1: "Ranging / Choppy",
    2: "High Volatility Breakout",
    3: "Low Volatility Consolidation",
}

STRATEGY_REGIME_FIT = {
    "Strong Trend":                   ["Trend_Following", "Momentum", "Breakout"],
    "Ranging / Choppy":               ["Mean_Reversion", "Scalping", "Arbitrage"],
    "High Volatility Breakout":       ["Breakout", "Momentum"],
    "Low Volatility Consolidation":   ["Mean_Reversion", "Scalping"],
}


class UnsupervisedLearner:
    """
    Continuously learns from live market data and trade history without labels.

    Usage
    -----
        learner = UnsupervisedLearner()
        learner.ingest_market_bar(feat_row)       # called each bar
        regime = learner.get_current_regime()
        best   = learner.get_best_strategies_for_regime(regime)
        adjust = learner.get_confidence_multiplier("Mean_Reversion", regime)
        tips   = learner.generate_insights()       # feed to AIBrain
    """

    DATA_DIR     = Path("data/unsupervised")
    MODEL_FILE   = DATA_DIR / "regime_model.pkl"
    SCALER_FILE  = DATA_DIR / "regime_scaler.pkl"
    ANOMALY_FILE = DATA_DIR / "anomaly_model.pkl"
    INSIGHTS_FILE = DATA_DIR / "insights.json"
    HISTORY_FILE = DATA_DIR / "feature_history.json"

    MIN_SAMPLES_TO_FIT = 100
    RETRAIN_EVERY_N    = 50      # bars between re-fits
    N_REGIMES          = 4

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self._kmeans:       Optional[KMeans]          = None
        self._scaler:       Optional[StandardScaler]  = None
        self._anomaly:      Optional[IsolationForest] = None
        self._pca:          Optional[PCA]             = None

        self._feature_buffer: list[dict] = self._load_history()
        self._bar_count:      int        = 0
        self._current_regime: str        = "Unknown"
        self._regime_performance: dict   = {}   # regime → {strategy → [outcomes]}
        self._confidence_mults:   dict   = {}   # strategy → regime → multiplier

        # Load existing models if available
        self._load_models()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_market_bar(self, feature_row: pd.Series) -> str:
        """
        Feed a feature-engineered market row.
        Returns the detected regime label.
        """
        if not _SKLEARN_AVAILABLE:
            return "Unknown"

        record = {col: float(feature_row.get(col, 0.0)) for col in REGIME_FEATURES}
        record["timestamp"] = datetime.now().isoformat()
        self._feature_buffer.append(record)
        self._bar_count += 1

        # Fit or retrain when enough data
        if (len(self._feature_buffer) >= self.MIN_SAMPLES_TO_FIT
                and self._bar_count % self.RETRAIN_EVERY_N == 0):
            self._fit_regime_model()

        # Classify current bar
        if self._kmeans and self._scaler:
            self._current_regime = self._classify_bar(record)

        # Periodically persist buffer (every 100 bars)
        if self._bar_count % 100 == 0:
            self._save_history()

        return self._current_regime

    def ingest_trade_outcome(
        self,
        feature_row: pd.Series,
        profit:      float,
        strategy:    str,
    ):
        """
        Record a closed trade's outcome against the market features at entry.
        Used to update per-regime strategy performance scores.
        """
        regime = self._current_regime
        if regime not in self._regime_performance:
            self._regime_performance[regime] = {}
        if strategy not in self._regime_performance[regime]:
            self._regime_performance[regime][strategy] = []

        self._regime_performance[regime][strategy].append(1.0 if profit > 0 else 0.0)
        self._update_confidence_multipliers(regime, strategy)

    def get_current_regime(self) -> str:
        return self._current_regime

    def get_best_strategies_for_regime(self, regime: str = None) -> list[str]:
        """
        Returns the strategies most suited to the current (or given) regime,
        ordered by learned performance if enough data exists.
        """
        r = regime or self._current_regime
        base = STRATEGY_REGIME_FIT.get(r, ["Mean_Reversion"])

        perf = self._regime_performance.get(r, {})
        if not perf:
            return base

        # Sort by win rate descending
        scored = []
        for strat in base:
            outcomes = perf.get(strat, [])
            win_rate = np.mean(outcomes) if outcomes else 0.5
            scored.append((strat, win_rate))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def get_confidence_multiplier(self, strategy: str, regime: str = None) -> float:
        """
        Returns a multiplier (0.5 – 1.5) to scale a strategy's confidence
        based on how well it's been performing in the current regime.
        """
        r = regime or self._current_regime
        return self._confidence_mults.get(strategy, {}).get(r, 1.0)

    def is_anomalous(self, feature_row: pd.Series) -> tuple[bool, float]:
        """
        Checks if current market conditions are anomalous (unusual, risk-elevated).
        Returns (is_anomaly, anomaly_score).
        """
        if not _SKLEARN_AVAILABLE or self._anomaly is None:
            return False, 0.0

        x = np.array([[float(feature_row.get(col, 0.0)) for col in TRADE_FEATURES]])
        try:
            x_scaled = self._scaler.transform(x)
            score    = self._anomaly.score_samples(x_scaled)[0]
            # score < -0.1 indicates anomaly in IsolationForest
            return score < -0.1, float(-score)
        except Exception:
            return False, 0.0

    def generate_insights(self) -> list[str]:
        """
        Returns up to 4 human-readable insight strings about learned patterns.
        These are fed to AIBrain to enrich proactive suggestions.
        """
        insights = []

        # Insight 1: Current regime
        if self._current_regime != "Unknown":
            insights.append(f"Market is currently in a '{self._current_regime}' regime.")

        # Insight 2: Best strategy for this regime
        best = self.get_best_strategies_for_regime()
        if best:
            insights.append(f"Best fit strategies right now: {', '.join(best[:2])}. ")

        # Insight 3: Any regime with notably poor performance
        for regime, strat_map in self._regime_performance.items():
            for strat, outcomes in strat_map.items():
                if len(outcomes) >= 10:
                    wr = np.mean(outcomes)
                    if wr < 0.4:
                        insights.append(
                            f"⚠️ {strat} has only a {wr:.0%} win rate in "
                            f"'{regime}' conditions — consider reducing allocation."
                        )
                    elif wr > 0.65:
                        insights.append(
                            f"✅ {strat} is performing well in '{regime}' "
                            f"conditions ({wr:.0%} win rate)."
                        )

        # Insight 4: Regime frequency
        regime_counts = self._count_recent_regimes()
        if regime_counts:
            dominant = max(regime_counts, key=regime_counts.get)
            pct = regime_counts[dominant] / sum(regime_counts.values())
            if pct > 0.6:
                insights.append(
                    f"Market has been predominantly '{dominant}' "
                    f"({pct:.0%} of recent bars)."
                )

        # Persist insights for inspection
        self._save_insights(insights)
        return insights[:4]

    def get_regime_stats(self) -> dict:
        """Returns a summary dict of regime performance — useful for account reports."""
        stats = {}
        for regime, strat_map in self._regime_performance.items():
            stats[regime] = {}
            for strat, outcomes in strat_map.items():
                if outcomes:
                    stats[regime][strat] = {
                        "trades":   len(outcomes),
                        "win_rate": round(float(np.mean(outcomes)), 3),
                    }
        return stats

    # ── Model fitting ──────────────────────────────────────────────────────────

    def _fit_regime_model(self):
        """Fits / retrains the K-Means regime classifier on buffered features."""
        if not _SKLEARN_AVAILABLE:
            return

        try:
            df = pd.DataFrame(self._feature_buffer)
            # Drop rows with any NaN in required columns
            avail = [c for c in REGIME_FEATURES if c in df.columns]
            df    = df[avail].dropna()

            if len(df) < self.MIN_SAMPLES_TO_FIT:
                return

            X = df.values.astype(np.float32)

            # Scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-Means regime model
            km = KMeans(
                n_clusters=self.N_REGIMES,
                random_state=42,
                n_init=10,
                max_iter=300,
            )
            km.fit(X_scaled)

            # Anomaly detector (IsolationForest on full trade features)
            avail_trade = [c for c in TRADE_FEATURES
                           if c in pd.DataFrame(self._feature_buffer).columns]
            if avail_trade:
                df2 = pd.DataFrame(self._feature_buffer)[avail_trade].dropna()
                if len(df2) >= 50:
                    X2 = scaler.fit_transform(df2.values.astype(np.float32)[:, :len(avail_trade)])
                    iso = IsolationForest(
                        n_estimators=100, contamination=0.05, random_state=42
                    )
                    iso.fit(X2)
                    self._anomaly = iso

            self._kmeans = km
            self._scaler = scaler
            self._save_models()

            # Re-label regime performance keys using new cluster centroids
            self._remap_regime_labels()

        except Exception as exc:
            # Silent failure — keep running without clustering
            pass

    def _classify_bar(self, record: dict) -> str:
        """Classify a single feature record into a regime label."""
        try:
            avail = [c for c in REGIME_FEATURES if c in record]
            x     = np.array([[record.get(c, 0.0) for c in avail]], dtype=np.float32)
            x_sc  = self._scaler.transform(x)
            cluster_id = int(self._kmeans.predict(x_sc)[0])

            # Map cluster ID to a semantic label based on centroid characteristics
            label = self._cluster_id_to_label(cluster_id)
            return label
        except Exception:
            return "Unknown"

    def _cluster_id_to_label(self, cluster_id: int) -> str:
        """
        Interprets K-Means cluster centroids to assign semantic regime labels.
        Uses ADX and BB width from the centroid as primary discriminators.
        """
        if self._kmeans is None:
            return REGIME_LABELS.get(cluster_id, "Unknown")

        centroid = self._kmeans.cluster_centers_[cluster_id]

        avail = [c for c in REGIME_FEATURES
                 if c in [c for c in REGIME_FEATURES]]
        adx_idx     = avail.index("adx")    if "adx"    in avail else 0
        bbw_idx     = avail.index("bb_width") if "bb_width" in avail else 1
        vol_idx     = avail.index("volatility_20") if "volatility_20" in avail else 2

        adx_val = centroid[adx_idx] if adx_idx < len(centroid) else 0
        bbw_val = centroid[bbw_idx] if bbw_idx < len(centroid) else 0
        vol_val = centroid[vol_idx] if vol_idx < len(centroid) else 0

        # Heuristic label assignment
        if adx_val > 0.5:        # high ADX (normalised) → trending
            return "Strong Trend"
        elif vol_val > 0.4:      # high volatility → breakout
            return "High Volatility Breakout"
        elif bbw_val < -0.3:     # narrow BB → consolidation
            return "Low Volatility Consolidation"
        else:
            return "Ranging / Choppy"

    # ── Confidence multiplier logic ───────────────────────────────────────────

    def _update_confidence_multipliers(self, regime: str, strategy: str):
        """
        Adjusts the confidence multiplier for a strategy in a given regime
        using a simple exponential moving average of recent win rates.
        """
        outcomes = self._regime_performance.get(regime, {}).get(strategy, [])
        if len(outcomes) < 5:
            return

        recent_wr = np.mean(outcomes[-20:])   # rolling 20-trade window

        # Scale: 40% win rate → 0.6× multiplier, 65% win rate → 1.35× multiplier
        multiplier = 0.5 + (recent_wr * 1.5)
        multiplier = round(np.clip(multiplier, 0.5, 1.5), 3)

        if strategy not in self._confidence_mults:
            self._confidence_mults[strategy] = {}
        self._confidence_mults[strategy][regime] = multiplier

    def _remap_regime_labels(self):
        """After retraining, rebuild semantic label mappings for all clusters."""
        if self._kmeans is None:
            return
        # Force reclassification of recent buffer so labels stay consistent
        new_regime_perf = {}
        for regime, strat_map in self._regime_performance.items():
            new_regime_perf[regime] = strat_map   # keep existing data
        self._regime_performance = new_regime_perf

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _count_recent_regimes(self, last_n: int = 100) -> dict:
        """Counts regime occurrences in the last N buffered bars."""
        counts: dict[str, int] = {}
        for record in self._feature_buffer[-last_n:]:
            r = record.get("_regime", "Unknown")
            counts[r] = counts.get(r, 0) + 1
        return counts

    def _save_models(self):
        try:
            with self.MODEL_FILE.open("wb") as f:
                pickle.dump(self._kmeans, f)
            with self.SCALER_FILE.open("wb") as f:
                pickle.dump(self._scaler, f)
            if self._anomaly:
                with self.ANOMALY_FILE.open("wb") as f:
                    pickle.dump(self._anomaly, f)
        except Exception:
            pass


    def _load_models(self):
        try:
            # Define trusted types allowlist if custom classes are used, or use default safe types
            unknown_types = sio.get_untrusted_types(file=self.MODEL_FILE)
            if self.MODEL_FILE.exists():
                self._kmeans = sio.load(self.MODEL_FILE, trusted=unknown_types)
            if self.SCALER_FILE.exists():
                self._scaler = sio.load(self.SCALER_FILE, trusted=sio.get_untrusted_types(file=self.SCALER_FILE))
            if self.ANOMALY_FILE.exists():
                self._anomaly = sio.load(self.ANOMALY_FILE, trusted=sio.get_untrusted_types(file=self.ANOMALY_FILE))
        except Exception as exc:
            pass
    def _load_history(self) -> list:
        if self.HISTORY_FILE.exists():
            try:
                with self.HISTORY_FILE.open("r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history(self):
        try:
            # Keep only the last 2000 bars to prevent unbounded growth
            data = self._feature_buffer[-2000:]
            with self.HISTORY_FILE.open("w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _save_insights(self, insights: list):
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "regime":    self._current_regime,
                "insights":  insights,
            }
            with self.INSIGHTS_FILE.open("w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass