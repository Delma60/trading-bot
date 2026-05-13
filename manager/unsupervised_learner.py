"""
manager/unsupervised_learner.py — ARIA's Self-Discovery Engine

FIX #20: _save_models() used pickle; _load_models() used skops — incompatible.
Both now use pickle consistently so trained models survive restarts.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


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
    DATA_DIR      = Path("data/unsupervised")
    MODEL_FILE    = DATA_DIR / "regime_model.pkl"
    SCALER_FILE   = DATA_DIR / "regime_scaler.pkl"
    ANOMALY_FILE  = DATA_DIR / "anomaly_model.pkl"
    INSIGHTS_FILE = DATA_DIR / "insights.json"
    HISTORY_FILE  = DATA_DIR / "feature_history.json"

    MIN_SAMPLES_TO_FIT = 100
    RETRAIN_EVERY_N    = 50
    N_REGIMES          = 4

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self._kmeans:   Optional[KMeans]         = None
        self._scaler:   Optional[StandardScaler] = None
        self._anomaly:  Optional[IsolationForest]= None

        self._feature_buffer: list[dict] = self._load_history()
        self._bar_count:      int        = 0
        self._current_regime: str        = "Unknown"
        self._regime_performance: dict   = {}
        self._confidence_mults:   dict   = {}

        self._load_models()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_market_bar(self, feature_row: pd.Series) -> str:
        if not _SKLEARN_AVAILABLE:
            return "Unknown"

        record = {col: float(feature_row.get(col, 0.0)) for col in REGIME_FEATURES}
        record["timestamp"] = datetime.now().isoformat()
        self._feature_buffer.append(record)
        self._bar_count += 1

        if (len(self._feature_buffer) >= self.MIN_SAMPLES_TO_FIT
                and self._bar_count % self.RETRAIN_EVERY_N == 0):
            self._fit_regime_model()

        if self._kmeans and self._scaler:
            self._current_regime = self._classify_bar(record)

        if self._bar_count % 100 == 0:
            self._save_history()

        return self._current_regime

    def ingest_trade_outcome(self, feature_row: pd.Series, profit: float, strategy: str):
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
        r    = regime or self._current_regime
        base = STRATEGY_REGIME_FIT.get(r, ["Mean_Reversion"])
        perf = self._regime_performance.get(r, {})
        if not perf:
            return base
        scored = []
        for strat in base:
            outcomes = perf.get(strat, [])
            win_rate = np.mean(outcomes) if outcomes else 0.5
            scored.append((strat, win_rate))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def get_confidence_multiplier(self, strategy: str, regime: str = None) -> float:
        r = regime or self._current_regime
        return self._confidence_mults.get(strategy, {}).get(r, 1.0)

    def is_anomalous(self, feature_row: pd.Series) -> tuple[bool, float]:
        if not _SKLEARN_AVAILABLE or self._anomaly is None or self._scaler is None:
            return False, 0.0
        x = np.array([[float(feature_row.get(col, 0.0)) for col in TRADE_FEATURES]])
        try:
            x_scaled = self._scaler.transform(x)
            score    = self._anomaly.score_samples(x_scaled)[0]
            return score < -0.1, float(-score)
        except Exception:
            return False, 0.0

    def generate_insights(self) -> list[str]:
        insights = []
        if self._current_regime != "Unknown":
            insights.append(f"Market is currently in a '{self._current_regime}' regime.")
        best = self.get_best_strategies_for_regime()
        if best:
            insights.append(f"Best fit strategies right now: {', '.join(best[:2])}.")
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
        regime_counts = self._count_recent_regimes()
        if regime_counts:
            dominant = max(regime_counts, key=regime_counts.get)
            pct = regime_counts[dominant] / sum(regime_counts.values())
            if pct > 0.6:
                insights.append(
                    f"Market has been predominantly '{dominant}' "
                    f"({pct:.0%} of recent bars)."
                )
        self._save_insights(insights)
        return insights[:4]

    def get_regime_stats(self) -> dict:
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

    # ── Model fitting ─────────────────────────────────────────────────────────

    def _fit_regime_model(self):
        if not _SKLEARN_AVAILABLE:
            return
        try:
            df   = pd.DataFrame(self._feature_buffer)
            avail = [c for c in REGIME_FEATURES if c in df.columns]
            df   = df[avail].dropna()
            if len(df) < self.MIN_SAMPLES_TO_FIT:
                return
            X = df.values.astype(np.float32)
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            km = KMeans(n_clusters=self.N_REGIMES, random_state=42, n_init=10, max_iter=300)
            km.fit(X_scaled)

            avail_trade = [c for c in TRADE_FEATURES
                           if c in pd.DataFrame(self._feature_buffer).columns]
            anomaly = None
            if avail_trade:
                df2 = pd.DataFrame(self._feature_buffer)[avail_trade].dropna()
                if len(df2) >= 50:
                    X2 = scaler.fit_transform(df2.values.astype(np.float32)[:, :len(avail_trade)])
                    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                    iso.fit(X2)
                    anomaly = iso

            self._kmeans  = km
            self._scaler  = scaler
            self._anomaly = anomaly
            self._save_models()
            self._remap_regime_labels()
        except Exception:
            pass

    def _classify_bar(self, record: dict) -> str:
        try:
            avail = [c for c in REGIME_FEATURES if c in record]
            x     = np.array([[record.get(c, 0.0) for c in avail]], dtype=np.float32)
            x_sc  = self._scaler.transform(x)
            cluster_id = int(self._kmeans.predict(x_sc)[0])
            return self._cluster_id_to_label(cluster_id)
        except Exception:
            return "Unknown"

    def _cluster_id_to_label(self, cluster_id: int) -> str:
        if self._kmeans is None:
            return REGIME_LABELS.get(cluster_id, "Unknown")
        centroid = self._kmeans.cluster_centers_[cluster_id]
        avail    = [c for c in REGIME_FEATURES]
        adx_idx  = avail.index("adx")     if "adx"     in avail else 0
        bbw_idx  = avail.index("bb_width") if "bb_width" in avail else 1
        vol_idx  = avail.index("volatility_20") if "volatility_20" in avail else 2
        adx_val  = centroid[adx_idx] if adx_idx < len(centroid) else 0
        bbw_val  = centroid[bbw_idx] if bbw_idx < len(centroid) else 0
        vol_val  = centroid[vol_idx] if vol_idx < len(centroid) else 0
        if adx_val > 0.5:   return "Strong Trend"
        elif vol_val > 0.4: return "High Volatility Breakout"
        elif bbw_val < -0.3:return "Low Volatility Consolidation"
        return "Ranging / Choppy"

    # ── Confidence multiplier ─────────────────────────────────────────────────

    def _update_confidence_multipliers(self, regime: str, strategy: str):
        outcomes = self._regime_performance.get(regime, {}).get(strategy, [])
        if len(outcomes) < 5:
            return
        recent_wr  = np.mean(outcomes[-20:])
        multiplier = round(np.clip(0.5 + (recent_wr * 1.5), 0.5, 1.5), 3)
        if strategy not in self._confidence_mults:
            self._confidence_mults[strategy] = {}
        self._confidence_mults[strategy][regime] = multiplier

    def _remap_regime_labels(self):
        if self._kmeans is None:
            return
        # Keep existing performance data; labels stay consistent
        self._regime_performance = dict(self._regime_performance)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_models(self):
        """
        FIX #20: Save ALL models with pickle so _load_models() (also pickle)
        can round-trip them correctly.  Previously saved with pickle but
        loaded with skops — an incompatible pair.
        """
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
        """
        FIX #20: Load with pickle to match _save_models().
        The old code loaded with skops while saving with pickle, so
        the learner always started from scratch after every restart.
        """
        try:
            if self.MODEL_FILE.exists():
                with self.MODEL_FILE.open("rb") as f:
                    self._kmeans = pickle.load(f)
            if self.SCALER_FILE.exists():
                with self.SCALER_FILE.open("rb") as f:
                    self._scaler = pickle.load(f)
            if self.ANOMALY_FILE.exists():
                with self.ANOMALY_FILE.open("rb") as f:
                    self._anomaly = pickle.load(f)
        except Exception:
            # Corrupt or incompatible saved model — start fresh
            self._kmeans  = None
            self._scaler  = None
            self._anomaly = None

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

    def _count_recent_regimes(self, last_n: int = 100) -> dict:
        counts: dict[str, int] = {}
        for record in self._feature_buffer[-last_n:]:
            r = record.get("_regime", "Unknown")
            counts[r] = counts.get(r, 0) + 1
        return counts