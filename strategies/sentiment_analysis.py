"""
strategies/sentiment_analysis.py — Market Sentiment Analysis Strategy

Measures the internal "mood" of price action using a multi-layer
sentiment model built entirely from OHLCV data. No external feeds
required — all signals are derived from price structure, volume
behaviour, and statistical momentum divergence.

Architecture (three sentiment layers)
--------------------------------------
Layer 1 — Price Action Sentiment
    Reads candlestick psychology: body ratios, shadow imbalance,
    engulfing patterns, and close position within the session range.
    Answers: "What are buyers/sellers doing *right now*?"

Layer 2 — Volume Sentiment
    Detects whether smart-money volume confirms or diverges from price.
    Uses OBV trend slope, volume-weighted price (VWAP) deviation, and
    selling/buying pressure ratios from candle construction.
    Answers: "Is money *flowing in* or being *distributed out*?"

Layer 3 — Statistical Momentum Sentiment
    RSI multi-period divergence, MACD histogram acceleration, and a
    custom Fear/Greed oscillator built from volatility and range
    expansion. Answers: "Is the crowd *accelerating* or *exhausting*?"

Scoring
-------
Each layer outputs a [-1.0, +1.0] sentiment score.
The final signal blends all three with configurable weights, producing
a continuous Composite Sentiment Score (CSS). A trade is only signalled
when CSS exceeds a threshold AND at least two layers agree on direction.

This plugs directly into the MetaScorer ensemble via the existing
check_signals() pipeline in strategy_manager.py.

Integration
-----------
    # In strategy_manager.py __init__:
    from strategies.sentiment_analysis import SentimentAnalysisStrategy

    self.engines["Sentiment_Analysis"] = SentimentAnalysisStrategy()

    # Also add to active_ensemble_strategies:
    self.active_ensemble_strategies = [
        k for k in self.engines
        if k not in ("News_Trading",)   # <-- remove "Sentiment_Analysis" from exclusion
    ]
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class SentimentAnalysisStrategy:
    """
    Multi-layer price-action sentiment analyser.

    Parameters
    ----------
    layer_weights : tuple[float, float, float]
        Relative weights for (price_action, volume, momentum) layers.
        Defaults to equal weighting (1/3 each).
    signal_threshold : float
        Minimum absolute CSS needed to emit a BUY/SELL signal.
        Range 0.0–1.0. Higher = more selective.
    layer_agreement : int
        Minimum number of layers that must agree with the final direction
        (1, 2, or 3). Default 2 — two-layer consensus required.
    lookback_fast : int
        Short rolling window for sentiment smoothing.
    lookback_slow : int
        Long rolling window for trend/baseline comparison.
    fear_greed_period : int
        Bars for the Fear/Greed oscillator calculation.
    volume_period : int
        Bars for volume-based calculations (VWAP, OBV slope).
    min_bars : int
        Absolute minimum bars needed before any signal is emitted.
    """

    def __init__(
        self,
        layer_weights:    tuple[float, float, float] = (1.0, 1.0, 1.0),
        signal_threshold: float = 0.35,
        layer_agreement:  int   = 2,
        lookback_fast:    int   = 10,
        lookback_slow:    int   = 30,
        fear_greed_period: int  = 14,
        volume_period:    int   = 20,
        min_bars:         int   = 60,
    ):
        self.layer_weights     = layer_weights
        self.signal_threshold  = signal_threshold
        self.layer_agreement   = layer_agreement
        self.lookback_fast     = lookback_fast
        self.lookback_slow     = lookback_slow
        self.fear_greed_period = fear_greed_period
        self.volume_period     = volume_period
        self.min_bars          = min_bars

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Analyse price-action sentiment and return a trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered OHLCV data (output of FeatureEngineer.compute).
            Falls back gracefully if engineered columns are absent.
        symbol : str
            Ticker — used for logging/reason strings only.

        Returns
        -------
        dict with keys: action, confidence, reason
        """
        if df is None or len(df) < self.min_bars:
            return self._wait(
                f"Insufficient data — need {self.min_bars} bars, "
                f"got {len(df) if df is not None else 0}."
            )

        try:
            data = df.copy()
            data = self._ensure_volume(data)

            # ── Three sentiment layers ────────────────────────────────────────
            pa_score, pa_reason  = self._price_action_sentiment(data)
            vol_score, vol_reason = self._volume_sentiment(data)
            mom_score, mom_reason = self._momentum_sentiment(data)

            # ── Composite Sentiment Score ─────────────────────────────────────
            w = self.layer_weights
            w_total = sum(w) or 1.0
            css = (
                pa_score  * (w[0] / w_total)
                + vol_score * (w[1] / w_total)
                + mom_score * (w[2] / w_total)
            )
            css = float(np.clip(css, -1.0, 1.0))

            # ── Layer agreement check ─────────────────────────────────────────
            scores = [pa_score, vol_score, mom_score]
            direction = "BUY" if css > 0 else "SELL"
            agreeing = sum(1 for s in scores if (s > 0) == (css > 0))

            if abs(css) < self.signal_threshold:
                return self._wait(
                    f"CSS {css:+.3f} below threshold {self.signal_threshold:.2f}. "
                    f"Sentiment neutral — no trade. "
                    f"PA={pa_score:+.2f} Vol={vol_score:+.2f} Mom={mom_score:+.2f}."
                )

            if agreeing < self.layer_agreement:
                return self._wait(
                    f"CSS {css:+.3f} passes threshold but only {agreeing}/3 layers "
                    f"agree (need {self.layer_agreement}). "
                    f"PA={pa_score:+.2f} Vol={vol_score:+.2f} Mom={mom_score:+.2f}."
                )

            # ── Confidence calibration ────────────────────────────────────────
            # Scale from threshold→1.0 into the 0.50→0.95 output range
            raw_strength = (abs(css) - self.signal_threshold) / (1.0 - self.signal_threshold + 1e-9)
            agreement_bonus = (agreeing - self.layer_agreement) * 0.05
            confidence = round(min(0.50 + raw_strength * 0.45 + agreement_bonus, 0.95), 2)

            sym_str = f" [{symbol}]" if symbol else ""
            reason = (
                f"Sentiment{sym_str} CSS {css:+.3f} → {direction}. "
                f"Layers({agreeing}/3 agree): "
                f"PA={pa_score:+.2f} [{pa_reason}] | "
                f"Vol={vol_score:+.2f} [{vol_reason}] | "
                f"Mom={mom_score:+.2f} [{mom_reason}]."
            )

            return {
                "action":     direction,
                "confidence": confidence,
                "reason":     reason,
                # Extra context for logging/debugging
                "css":        round(css, 4),
                "pa_score":   round(pa_score, 3),
                "vol_score":  round(vol_score, 3),
                "mom_score":  round(mom_score, 3),
            }

        except Exception as exc:
            return self._wait(f"Sentiment analysis error: {exc}")

    # ── Layer 1: Price Action Sentiment ──────────────────────────────────────

    def _price_action_sentiment(self, df: pd.DataFrame) -> tuple[float, str]:
        """
        Reads candlestick psychology over the recent lookback window.

        Factors
        -------
        - Body ratio:    large bodies = conviction; small = indecision
        - Shadow bias:   upper shadow > lower = bearish rejection; vice versa
        - Close position: close in top/bottom of bar (0=bottom, 1=top)
        - Engulfing count: recent full-body reversals
        - Consecutive direction: how many bars of same-direction body
        """
        bars = df.iloc[-(self.lookback_fast + 5):-1]  # last confirmed closed bars

        if len(bars) < 5:
            return 0.0, "insufficient bars"

        open_  = bars["open"].values
        high   = bars["high"].values
        low    = bars["low"].values
        close  = bars["close"].values

        body        = close - open_
        bar_range   = (high - low) + 1e-9
        body_ratio  = np.abs(body) / bar_range          # 0=doji, 1=marubozu
        upper_shadow = high - np.maximum(close, open_)
        lower_shadow = np.minimum(close, open_) - low
        shadow_bias  = (lower_shadow - upper_shadow) / bar_range   # +1=bull, -1=bear
        close_pos    = (close - low) / bar_range                   # 0=bottom, 1=top

        # Weighted recency: more recent bars matter more
        n      = len(bars)
        recency_weights = np.linspace(0.5, 1.0, n)
        w_sum  = recency_weights.sum()

        # Body direction score (positive bodies = bullish)
        direction_scores = np.sign(body) * body_ratio
        body_score = float(np.dot(direction_scores, recency_weights) / w_sum)

        # Shadow sentiment: lower shadow > upper = accumulation (bullish)
        shadow_score = float(np.dot(shadow_bias, recency_weights) / w_sum)

        # Close position: >0.5 = bullish (closed near high)
        close_pos_score = float(np.dot((close_pos * 2 - 1), recency_weights) / w_sum)

        # Engulfing pattern count (quick scan of last 5 bars)
        engulf_score = 0.0
        for i in range(1, min(5, n)):
            prev_body = abs(body[-(i+1)])
            curr_body = abs(body[-i])
            if curr_body > prev_body * 1.1:
                engulf_score += np.sign(body[-i]) * 0.2

        # Blend
        pa_raw = (
            body_score     * 0.40
            + shadow_score   * 0.25
            + close_pos_score * 0.25
            + np.clip(engulf_score, -0.3, 0.3) * 0.10
        )
        pa_score = float(np.clip(pa_raw, -1.0, 1.0))

        # Human reason
        dominant = "bullish" if pa_score > 0 else "bearish"
        intensity = "strong" if abs(pa_score) > 0.5 else "mild"
        reason = f"{intensity} {dominant} candles"

        return pa_score, reason

    # ── Layer 2: Volume Sentiment ─────────────────────────────────────────────

    def _volume_sentiment(self, df: pd.DataFrame) -> tuple[float, str]:
        """
        Evaluates money flow using OBV slope, VWAP deviation, and
        buying/selling pressure derived from bar construction.

        Buying pressure  = lower shadow / bar range (accumulation wicks)
        Selling pressure = upper shadow / bar range (distribution wicks)
        """
        n     = min(self.volume_period, len(df) - 1)
        bars  = df.iloc[-(n + 1):-1]

        if len(bars) < 10:
            return 0.0, "insufficient volume data"

        close  = bars["close"].values
        high   = bars["high"].values
        low    = bars["low"].values
        open_  = bars["open"].values
        volume = bars["volume"].values

        bar_range = (high - low) + 1e-9

        # ── OBV slope ────────────────────────────────────────────────────────
        price_change = np.diff(close, prepend=close[0])
        obv   = np.cumsum(np.sign(price_change) * volume)
        # Normalised slope (linear regression coefficient)
        x     = np.arange(len(obv))
        if x.std() > 0:
            obv_slope = float(np.corrcoef(x, obv)[0, 1])   # -1 to +1
        else:
            obv_slope = 0.0

        # ── VWAP deviation ───────────────────────────────────────────────────
        # Positive deviation = price above VWAP = bullish sentiment
        typical_price = (high + low + close) / 3.0
        vwap = np.sum(typical_price * volume) / (np.sum(volume) + 1e-9)
        current_price = close[-1]
        price_range   = np.std(close) or 1.0
        vwap_dev      = float(np.clip((current_price - vwap) / price_range, -1.0, 1.0))

        # ── Buying vs Selling Pressure ────────────────────────────────────────
        # Bullish bar: body is bullish AND volume is above average
        # Bearish bar: body is bearish AND volume is above average
        body       = close - open_
        avg_volume = volume.mean()
        vol_factor = volume / (avg_volume + 1e-9)

        buy_pressure  = np.sum(np.where(body > 0, vol_factor, 0.0)) / (n + 1e-9)
        sell_pressure = np.sum(np.where(body < 0, vol_factor, 0.0)) / (n + 1e-9)
        total_pressure = buy_pressure + sell_pressure + 1e-9
        pressure_score = float((buy_pressure - sell_pressure) / total_pressure)

        # ── Climax volume detection ───────────────────────────────────────────
        # Very high volume on a down bar (potential exhaustion / reversal)
        max_vol_idx = np.argmax(volume)
        climax_sign = np.sign(body[max_vol_idx])
        climax_ratio = volume[max_vol_idx] / (avg_volume + 1e-9)
        climax_score = 0.0
        if climax_ratio > 2.5:
            # Climactic volume on a down bar = bullish (selling exhaustion)
            # Climactic volume on an up bar  = bearish (buying exhaustion)
            climax_score = float(-climax_sign * min(climax_ratio / 5.0, 0.3))

        # Blend
        vol_raw = (
            obv_slope     * 0.35
            + vwap_dev      * 0.30
            + pressure_score * 0.25
            + climax_score   * 0.10
        )
        vol_score = float(np.clip(vol_raw, -1.0, 1.0))

        dominant = "inflow" if vol_score > 0 else "outflow"
        reason   = f"volume {dominant} (OBV slope {obv_slope:+.2f})"
        return vol_score, reason

    # ── Layer 3: Momentum Sentiment ───────────────────────────────────────────

    def _momentum_sentiment(self, df: pd.DataFrame) -> tuple[float, str]:
        """
        Measures whether crowd momentum is accelerating, steady, or exhausting.

        Components
        ----------
        RSI multi-period divergence:
            Fast RSI > Slow RSI = momentum accelerating upward
        MACD histogram trend:
            Positive and growing = bullish; negative and falling = bearish
        Fear/Greed oscillator:
            Built from ATR expansion vs baseline.
            Expanding ATR + upward close = greed (BUY); contracting = fear (SELL)
        Consecutive higher closes:
            Simple price momentum persistence measure.
        """
        n    = min(self.fear_greed_period + self.lookback_slow, len(df) - 1)
        bars = df.iloc[-(n + 1):-1]

        if len(bars) < self.lookback_slow:
            return 0.0, "insufficient momentum data"

        close  = bars["close"].values
        high   = bars["high"].values
        low    = bars["low"].values

        # ── RSI (fast vs slow) ────────────────────────────────────────────────
        rsi_fast = self._rsi(close, self.lookback_fast)
        rsi_slow = self._rsi(close, self.lookback_slow)
        # Normalise both to [-1, +1] from [0, 100]
        # Divergence: fast > slow = accelerating bull; fast < slow = accelerating bear
        if rsi_fast is not None and rsi_slow is not None:
            rsi_div = float(np.clip((rsi_fast - rsi_slow) / 50.0, -1.0, 1.0))
            # Also absolute level: RSI > 55 = bullish bias, < 45 = bearish bias
            rsi_level = float(np.clip((rsi_fast - 50.0) / 50.0, -1.0, 1.0))
            rsi_score = rsi_div * 0.6 + rsi_level * 0.4
        else:
            rsi_score = 0.0

        # ── MACD histogram momentum ──────────────────────────────────────────
        macd_hist = self._macd_histogram(close)
        if macd_hist is not None and len(macd_hist) >= 3:
            last_hist   = macd_hist[-1]
            prev_hist   = macd_hist[-2]
            pp_hist     = macd_hist[-3]
            # Accelerating: histogram is growing in magnitude in the same direction
            accel = (last_hist - prev_hist) + (prev_hist - pp_hist)
            # Normalise by ATR-like spread of histogram
            hist_std = np.std(macd_hist[-self.lookback_fast:]) or 1.0
            macd_score = float(np.clip(last_hist / hist_std * 0.5 + accel / hist_std * 0.5, -1.0, 1.0))
        else:
            macd_score = 0.0

        # ── Fear/Greed oscillator ────────────────────────────────────────────
        # ATR expansion relative to its own average = "greed" when expanding with up move
        atr_period = self.fear_greed_period
        if len(close) > atr_period + 5:
            tr_vals  = np.maximum(high - low,
                       np.maximum(
                           np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:]  - close[:-1])
                       ) if len(close) > 1 else high - low)
            # Pad tr_vals to match length
            tr_vals = np.concatenate([[tr_vals[0]], tr_vals])[:len(close)]
            atr_now  = float(np.mean(tr_vals[-atr_period:]))
            atr_base = float(np.mean(tr_vals[-atr_period * 2: -atr_period])) if len(tr_vals) >= atr_period * 2 else atr_now
            atr_expansion = (atr_now - atr_base) / (atr_base + 1e-9)   # >0 = expanding volatility

            # Combine ATR expansion with price direction
            recent_return = float((close[-1] - close[-atr_period]) / (close[-atr_period] + 1e-9))
            fg_raw = float(np.clip(recent_return * (1.0 + atr_expansion), -1.0, 1.0))
            fg_score = fg_raw
        else:
            fg_score = 0.0

        # ── Consecutive direction count ───────────────────────────────────────
        last_close   = close[-1]
        consec_score = 0.0
        count        = 0
        for i in range(2, min(8, len(close))):
            if close[-i] < close[-i+1]:
                count += 1
            else:
                break
        consec_bull = count / 7.0
        count = 0
        for i in range(2, min(8, len(close))):
            if close[-i] > close[-i+1]:
                count += 1
            else:
                break
        consec_bear = count / 7.0
        consec_score = float(consec_bull - consec_bear)

        # Blend
        mom_raw = (
            rsi_score    * 0.35
            + macd_score   * 0.30
            + fg_score     * 0.25
            + consec_score * 0.10
        )
        mom_score = float(np.clip(mom_raw, -1.0, 1.0))

        dominant = "bullish" if mom_score > 0 else "bearish"
        reason   = f"{dominant} momentum (RSI div {rsi_score:+.2f}, FG {fg_score:+.2f})"
        return mom_score, reason

    # ── Technical helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> Optional[float]:
        """Compute RSI for the last bar only."""
        if len(close) < period + 1:
            return None
        delta = np.diff(close[-(period + 1):])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[np.ndarray]:
        """Compute MACD histogram series. Returns None if not enough data."""
        if len(close) < slow + signal:
            return None

        def ema(arr, span):
            alpha = 2.0 / (span + 1)
            result = np.empty_like(arr)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast   = ema(close, fast)
        ema_slow   = ema(close, slow)
        macd_line  = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        return macd_line - signal_line

    @staticmethod
    def _ensure_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee a 'volume' column exists, falling back to tick/real volume."""
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df = df.copy()
                df["volume"] = df["tick_volume"]
            elif "real_volume" in df.columns:
                df = df.copy()
                df["volume"] = df["real_volume"]
            else:
                df = df.copy()
                df["volume"] = 1.0   # flat volume fallback — volume layer will be muted
        return df

    @staticmethod
    def _wait(reason: str) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": reason}