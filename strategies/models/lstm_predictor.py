import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Features the LSTM uses — must exist in the feature-engineered DataFrame.
# Chosen for being normalisation-friendly and carrying directional signal.
LSTM_FEATURE_COLS = [
    "return_1", "return_3", "return_5", "log_return",
    "volatility_10", "volatility_20", "atr_ratio",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi_14", "rsi_7",
    "macd_hist", "macd_cross",
    "stoch_k", "stoch_d",
    "williams_r", "cci",
    "adx", "di_diff",
    "bb_width", "bb_pos",
    "volume_ratio", "obv_trend",
    "dist_vwap",
    "body_ratio", "is_bullish",
    "regime_trending", "regime_bullish",
]


# ── Architecture ─────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """
    Stacked LSTM with LayerNorm + dropout, projecting to 3-class logits.
    Classes: 0=DOWN, 1=NEUTRAL, 2=UP
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out     = self.norm(out[:, -1, :])   # take the last time-step
        out     = self.dropout(out)
        return self.head(out)


# ── Public class ─────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Per-symbol LSTM that classifies the *next bar's* price direction.

    Lifecycle
    ---------
    1. Call ``train(feat_df, symbol)`` the first time — weights are saved to disk.
    2. On subsequent runs the weights are loaded automatically (no re-training).
    3. Call ``predict(feat_df, symbol)`` to get a direction + confidence.

    Use ``force=True`` in ``train()`` to retrain from scratch.

    Return schema from ``predict()``
    ----------------------------------
    {
        "direction":     "UP" | "DOWN" | "NEUTRAL",
        "confidence":    float,          # probability of the winning class
        "probabilities": {"DOWN": float, "NEUTRAL": float, "UP": float},
    }
    """

    CLASSES = ["DOWN", "NEUTRAL", "UP"]

    def __init__(
        self,
        seq_len:     int   = 30,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        lr:          float = 1e-3,
        epochs:      int   = 30,
        batch_size:  int   = 64,
        threshold:   float = 0.001,   # ±0.1 % to be classed as directional
        weights_dir: str   = "models/weights",
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.threshold   = threshold
        self.weights_dir = weights_dir
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # In-memory caches — keyed by symbol
        self._models:  dict[str, _LSTMNet]       = {}
        self._scalers: dict[str, StandardScaler] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str, force: bool = False) -> None:
        """Train the LSTM for ``symbol``, or load cached weights if they exist."""
        mp, sp = self._paths(symbol)

        if not force and os.path.exists(mp) and os.path.exists(sp):
            self._load(symbol, mp, sp)
            return

        X, y, scaler = self._build_sequences(df)
        if X is None:
            return

        n_features = X.shape[2]
        model = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Inverse-frequency class weighting handles the neutral-heavy distribution
        counts  = np.bincount(y, minlength=3).astype(float)
        weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.LongTensor(y).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss / len(loader))

        # Persist to disk
        os.makedirs(self.weights_dir, exist_ok=True)
        torch.save(model.state_dict(), mp)
        with open(sp, "wb") as f:
            pickle.dump(scaler, f)

        model.eval()
        self._models[symbol]  = model
        self._scalers[symbol] = scaler

    def predict(self, df: pd.DataFrame, symbol: str) -> dict:
        """Return a direction prediction for the most recent bar."""
        if symbol not in self._models:
            return {"direction": "NEUTRAL", "confidence": 0.0, "probabilities": {}}

        X, _, _ = self._build_sequences(df, scaler=self._scalers[symbol], fit=False)
        if X is None or len(X) == 0:
            return {"direction": "NEUTRAL", "confidence": 0.0, "probabilities": {}}

        model = self._models[symbol]
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X[-1:]).to(self.device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx       = int(np.argmax(probs))
        direction = self.CLASSES[idx]
        return {
            "direction":     direction,
            "confidence":    float(probs[idx]),
            "probabilities": dict(zip(self.CLASSES, probs.tolist())),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_sequences(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler | None = None,
        fit: bool = True,
    ):
        """Return (X, y, scaler) — or (None, None, None) if not enough data."""
        available = [c for c in LSTM_FEATURE_COLS if c in df.columns]
        if len(available) < 5:
            return None, None, None

        data = df[available].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
        data = scaler.fit_transform(data) if fit else scaler.transform(data)

        # Label: next-bar direction
        returns = df["close"].pct_change().shift(-1).values

        X, y = [], []
        for i in range(self.seq_len, len(data) - 1):
            X.append(data[i - self.seq_len : i])
            ret = returns[i]
            if   ret >  self.threshold: y.append(2)    # UP
            elif ret < -self.threshold: y.append(0)    # DOWN
            else:                        y.append(1)   # NEUTRAL

        if not X:
            return None, None, scaler

        return np.array(X), np.array(y), scaler

    def _paths(self, symbol: str) -> tuple[str, str]:
        safe = symbol.replace("/", "_")
        return (
            os.path.join(self.weights_dir, f"lstm_{safe}.pt"),
            os.path.join(self.weights_dir, f"scaler_{safe}.pkl"),
        )

    def _load(self, symbol: str, mp: str, sp: str) -> None:
        with open(sp, "rb") as f:
            scaler = pickle.load(f)

        n_features = scaler.n_features_in_
        model = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        model.load_state_dict(torch.load(mp, map_location=self.device, weights_only=True))
        model.eval()

        self._models[symbol]  = model
        self._scalers[symbol] = scaler