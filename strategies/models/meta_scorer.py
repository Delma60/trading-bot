import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# XGBoost is optional — we fall back to pure weighted voting if unavailable.
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────

ACTION_MAP = {"BUY": 1, "SELL": -1, "WAIT": 0}

LSTM_TO_ACTION = {"UP": "BUY", "DOWN": "SELL", "NEUTRAL": "WAIT"}

# Ordered list so the feature vector is always assembled in the same order.
STRATEGY_NAMES = [
    "Mean_Reversion", "Momentum", "Breakout",
    "Scalping", "News_Trading", "Sentiment_Analysis", "Arbitrage",
]

# Market-context features included in the meta-vector.
CONTEXT_COLS = [
    "adx", "rsi_14", "volatility_20", "volume_ratio",
    "bb_pos", "bb_width", "regime_trending", "regime_bullish",
    "dist_sma50", "dist_sma200", "macd_hist", "di_diff",
]


# ── Public class ─────────────────────────────────────────────────────────────

class MetaScorer:
    """
    A meta-model (XGBoost) that learns which combination of strategy signals,
    LSTM predictions, and market-context features reliably leads to profitable
    trades.

    Feature vector per bar
    ----------------------
    For each of N strategies  →  [encoded_action, confidence]   (2N values)
    LSTM prediction           →  [encoded_direction, confidence] (2 values)
    Market context            →  CONTEXT_COLS                   (len(CONTEXT_COLS) values)

    Output
    ------
    {"action": "BUY"|"SELL"|"WAIT", "confidence": float,
     "probabilities": dict, "source": "meta_model"|"weighted_vote"}

    Training
    --------
    Collect labelled samples with ``collect_sample()`` during live/paper trading,
    then call ``train()`` once you have a few hundred samples.  Until then the
    scorer falls back to a calibrated weighted-vote across all signals.
    """

    def __init__(self, model_path: str = "models/weights/meta_scorer.pkl"):
        self.model_path = model_path
        self.model      = None
        self._samples: list[tuple[np.ndarray, str]] = []   # (feature_vec, label)
        self._load_if_exists()

    # ── Feature assembly ─────────────────────────────────────────────────────

    def build_feature_vector(
        self,
        strategy_signals: dict,   # {name: {"action": str, "confidence": float}}
        lstm_prediction:  dict,   # {"direction": str, "confidence": float}
        market_row:       pd.Series,
    ) -> np.ndarray:
        """Assemble the fixed-length feature vector the meta-model expects."""
        vec: list[float] = []

        # Strategy signals (action + confidence per strategy, in fixed order)
        for name in STRATEGY_NAMES:
            sig    = strategy_signals.get(name, {})
            action = ACTION_MAP.get(sig.get("action", "WAIT"), 0)
            conf   = float(sig.get("confidence", 0.0))
            vec.extend([action, conf])

        # LSTM signal
        lstm_action = LSTM_TO_ACTION.get(lstm_prediction.get("direction", "NEUTRAL"), "WAIT")
        vec.extend([
            ACTION_MAP.get(lstm_action, 0),
            float(lstm_prediction.get("confidence", 0.0)),
        ])

        # Market context features
        for col in CONTEXT_COLS:
            vec.append(float(market_row.get(col, 0.0)))

        return np.array(vec, dtype=np.float32).reshape(1, -1)

    # ── Inference ────────────────────────────────────────────────────────────

    def score(
        self,
        strategy_signals: dict,
        lstm_prediction:  dict,
        market_row:       pd.Series,
    ) -> dict:
        """
        Return the final trading decision.
        Uses the trained meta-model when available; weighted vote otherwise.
        """
        if self.model is not None:
            return self._model_score(strategy_signals, lstm_prediction, market_row)
        return self._weighted_vote(strategy_signals, lstm_prediction)

    # ── Training ─────────────────────────────────────────────────────────────

    def collect_sample(self, feature_vector: np.ndarray, label: str) -> None:
        """
        Accumulate a labelled training sample.
        Call this after each bar closes and you know whether the trade was
        profitable (label="BUY"/"SELL") or should have been skipped (label="WAIT").
        """
        self._samples.append((feature_vector.flatten(), label))

    def train(self, force: bool = False, min_samples: int = 200) -> float:
        """
        Train the XGBoost meta-model on collected samples.
        Returns validation accuracy, or 0.0 if training was skipped.

        You can also call this with pre-built arrays:
            scorer.train_on_arrays(X, y)
        """
        if not _XGB_AVAILABLE:
            print("[MetaScorer] xgboost not installed — staying in weighted-vote mode.")
            return 0.0

        if not force and self.model is not None:
            return 0.0

        if len(self._samples) < min_samples:
            print(f"[MetaScorer] Only {len(self._samples)} samples — need {min_samples} to train.")
            return 0.0

        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        return self._fit(X, y)

    def train_on_arrays(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train directly from pre-built feature matrix + label array."""
        if not _XGB_AVAILABLE:
            return 0.0
        return self._fit(X, y)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fit(self, X: np.ndarray, y: np.ndarray) -> float:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        self._save()
        print(f"[MetaScorer] Trained — val accuracy: {val_acc:.3f}")
        return float(val_acc)

    def _model_score(
        self,
        strategy_signals: dict,
        lstm_prediction:  dict,
        market_row:       pd.Series,
    ) -> dict:
        X       = self.build_feature_vector(strategy_signals, lstm_prediction, market_row)
        probs   = self.model.predict_proba(X)[0]
        labels  = list(self.model.classes_)
        idx     = int(np.argmax(probs))
        return {
            "action":        labels[idx],
            "confidence":    float(probs[idx]),
            "probabilities": dict(zip(labels, probs.tolist())),
            "source":        "meta_model",
        }

    def _weighted_vote(self, strategy_signals: dict, lstm_prediction: dict) -> dict:
        """
        Calibrated weighted majority vote used before the meta-model is trained.

        Weights
        -------
        - Each real strategy (non-Dummy):  1.0 × its own reported confidence
        - LSTM prediction:                 2.0 × its confidence (higher weight)
        - Dummy / WAIT strategies:         0.3 × confidence (discounted)
        """
        votes: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "WAIT": 0.0}

        dummy_names = {"News_Trading", "Sentiment_Analysis"}

        for name, sig in strategy_signals.items():
            action = sig.get("action", "WAIT")
            conf   = float(sig.get("confidence", 0.5))
            weight = 0.3 if name in dummy_names else 1.0
            if action in votes:
                votes[action] += conf * weight

        # LSTM gets double weight — it sees the whole sequence
        lstm_action = LSTM_TO_ACTION.get(lstm_prediction.get("direction", "NEUTRAL"), "WAIT")
        lstm_conf   = float(lstm_prediction.get("confidence", 0.5))
        votes[lstm_action] += lstm_conf * 2.0

        total = sum(votes.values()) or 1.0
        norm  = {k: v / total for k, v in votes.items()}

        best = max(norm, key=norm.get)
        return {
            "action":        best,
            "confidence":    norm[best],
            "probabilities": norm,
            "source":        "weighted_vote",
        }

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def _load_if_exists(self) -> None:
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"[MetaScorer] Loaded trained model from {self.model_path}")