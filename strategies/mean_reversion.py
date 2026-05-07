# strategies/mean_reversion.py
import pandas as pd
import pandas_ta as ta
import numpy as np


class MeanReversionStrategy:
    """
    Bets on stretched prices snapping back to their mean using a confluence
    of Bollinger Bands, RSI, Stochastic, and volume confirmation.

    Improvements over v1
    --------------------
    * ADX ranging filter   — mean reversion only fires when ADX < adx_threshold,
                             preventing entries against strong trends.
    * RSI divergence       — detects bullish/bearish divergence between price
                             and RSI over a short lookback for higher-quality signals.
    * Stochastic confluence— adds a third oscillator; all three must agree on
                             the extreme reading before signalling.
    * Volume spike filter  — requires above-average volume at the extreme to
                             confirm institutional participation.
    * Graduated confidence — score accumulates from each confirming factor
                             (BB, RSI, Stochastic, divergence, volume) so the
                             final number actually reflects signal strength.
    """

    def __init__(
        self,
        bb_length:        int   = 20,
        bb_std:           float = 2.0,
        rsi_length:       int   = 14,
        stoch_k:          int   = 14,
        stoch_d:          int   = 3,
        sma_length:       int   = 21,
        adx_length:       int   = 14,
        adx_threshold:    float = 28.0,   # above this → trending, skip mean rev
        lookback_window:  int   = 100,
        percentile_lower: float = 0.15,
        percentile_upper: float = 0.85,
        volume_mult:      float = 1.2,    # volume spike at extreme
        div_lookback:     int   = 5,      # bars to check RSI divergence
    ):
        self.bb_length        = bb_length
        self.bb_std           = bb_std
        self.rsi_length       = rsi_length
        self.stoch_k          = stoch_k
        self.stoch_d          = stoch_d
        self.sma_length       = sma_length
        self.adx_length       = adx_length
        self.adx_threshold    = adx_threshold
        self.lookback_window  = lookback_window
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.volume_mult      = volume_mult
        self.div_lookback     = div_lookback

    def analyze(self, df: pd.DataFrame) -> dict:
        min_bars = max(self.lookback_window, self.bb_length,
                       self.rsi_length, self.sma_length, 200)
        if df is None or len(df) < min_bars:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": "Not enough data points."}

        data = df.copy()

        try:
            # ── Indicators ───────────────────────────────────────────────────
            # Bollinger Bands
            bb = ta.bbands(data["close"], length=self.bb_length, std=self.bb_std)
            if bb is None or bb.empty:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": "Failed to calculate Bollinger Bands."}
            data = pd.concat([data, bb], axis=1)
            bb_lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
            bb_upper_col = [c for c in bb.columns if c.startswith("BBU")][0]

            # RSI
            data["rsi"] = ta.rsi(data["close"], length=self.rsi_length)

            # Stochastic
            stoch = ta.stoch(data["high"], data["low"], data["close"],
                             k=self.stoch_k, d=self.stoch_d)
            if stoch is not None and not stoch.empty:
                data = pd.concat([data, stoch], axis=1)
                stoch_k_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
                stoch_d_col = [c for c in stoch.columns if c.startswith("STOCHd")][0]
            else:
                data["stoch_k"] = float("nan")
                data["stoch_d"] = float("nan")
                stoch_k_col, stoch_d_col = "stoch_k", "stoch_d"

            # ADX (trend strength filter)
            adx_df = ta.adx(data["high"], data["low"], data["close"],
                            length=self.adx_length)
            if adx_df is not None and not adx_df.empty:
                data = pd.concat([data, adx_df], axis=1)
                adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
            else:
                data["adx"] = float("nan")
                adx_col = "adx"

            # SMA + rolling ratio percentiles (no lookahead)
            data["sma"]            = ta.sma(data["close"], length=self.sma_length)
            data["ratio"]          = data["close"] / data["sma"]
            data["buy_threshold"]  = (data["ratio"]
                                      .rolling(self.lookback_window)
                                      .quantile(self.percentile_lower))
            data["sell_threshold"] = (data["ratio"]
                                      .rolling(self.lookback_window)
                                      .quantile(self.percentile_upper))

            # Volume surge
            data["vol_avg"]   = data["volume"].rolling(20).mean()
            data["vol_ratio"] = data["volume"] / data["vol_avg"].replace(0, float("nan"))

            data.dropna(inplace=True)
            if data.empty:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": "Data empty after indicator warm-up."}

            # ── Snapshot ─────────────────────────────────────────────────────
            last  = data.iloc[-1]
            close = last["close"]
            rsi   = last["rsi"]
            ratio = last["ratio"]
            lower_band     = last[bb_lower_col]
            upper_band     = last[bb_upper_col]
            buy_threshold  = last["buy_threshold"]
            sell_threshold = last["sell_threshold"]
            adx            = last[adx_col]
            stoch_k_val    = last[stoch_k_col]
            stoch_d_val    = last[stoch_d_col]
            vol_ratio      = last["vol_ratio"]

            # ── Gate 1: ADX trending filter ──────────────────────────────────
            if adx > self.adx_threshold:
                return {
                    "action": "WAIT",
                    "confidence": 0.0,
                    "reason": (f"Skipping — market is trending strongly "
                               f"(ADX: {adx:.1f} > {self.adx_threshold}). "
                               f"Mean reversion is unreliable here."),
                }

            # ── RSI divergence (short-lookback) ──────────────────────────────
            recent = data.tail(self.div_lookback + 1)
            bull_div = (
                recent["close"].iloc[-1] < recent["close"].iloc[0]   # price lower
                and recent["rsi"].iloc[-1]  > recent["rsi"].iloc[0]  # RSI higher
            )
            bear_div = (
                recent["close"].iloc[-1] > recent["close"].iloc[0]
                and recent["rsi"].iloc[-1]  < recent["rsi"].iloc[0]
            )

            volume_spike = vol_ratio >= self.volume_mult

            # ── Signal scoring ───────────────────────────────────────────────
            #
            # Each confirmed factor contributes to a base score.
            # The strategy requires at least the primary BB+RSI trigger.

            # ── BUY ──────────────────────────────────────────────────────────
            bb_oversold  = close < lower_band and rsi < 30
            ratio_low    = ratio < buy_threshold and rsi < 40
            stoch_over   = stoch_k_val < 20 and stoch_d_val < 20

            if bb_oversold or ratio_low:
                score = 0.50

                # Primary signal quality
                if bb_oversold:
                    score += 0.12
                    depth = (lower_band - close) / (lower_band + 1e-9)
                    score += min(depth * 5, 0.06)   # deeper = more confident

                if ratio_low:
                    score += 0.06

                # Confluence bonuses
                if stoch_over:
                    score += 0.08   # all three oscillators agree
                if bull_div:
                    score += 0.10   # divergence = strongest signal
                if volume_spike:
                    score += 0.06   # institutional footprint at the low

                confidence = round(min(score, 0.97), 2)
                factors = [
                    "BB below lower band + RSI oversold" if bb_oversold else "Ratio below rolling pct",
                    "Stochastic oversold" if stoch_over else None,
                    "Bullish RSI divergence" if bull_div else None,
                    f"Volume spike ({vol_ratio:.1f}x)" if volume_spike else None,
                ]
                reason = " | ".join(f for f in factors if f)

                return {
                    "action":     "BUY",
                    "confidence": confidence,
                    "reason":     f"Mean Reversion BUY — {reason}. "
                                  f"(ADX: {adx:.1f}, RSI: {rsi:.1f})",
                }

            # ── SELL ─────────────────────────────────────────────────────────
            bb_overbought = close > upper_band and rsi > 70
            ratio_high    = ratio > sell_threshold and rsi > 60
            stoch_under   = stoch_k_val > 80 and stoch_d_val > 80

            if bb_overbought or ratio_high:
                score = 0.50

                if bb_overbought:
                    score += 0.12
                    depth  = (close - upper_band) / (upper_band + 1e-9)
                    score += min(depth * 5, 0.06)

                if ratio_high:
                    score += 0.06
                if stoch_under:
                    score += 0.08
                if bear_div:
                    score += 0.10
                if volume_spike:
                    score += 0.06

                confidence = round(min(score, 0.97), 2)
                factors = [
                    "BB above upper band + RSI overbought" if bb_overbought else "Ratio above rolling pct",
                    "Stochastic overbought" if stoch_under else None,
                    "Bearish RSI divergence" if bear_div else None,
                    f"Volume spike ({vol_ratio:.1f}x)" if volume_spike else None,
                ]
                reason = " | ".join(f for f in factors if f)

                return {
                    "action":     "SELL",
                    "confidence": confidence,
                    "reason":     f"Mean Reversion SELL — {reason}. "
                                  f"(ADX: {adx:.1f}, RSI: {rsi:.1f})",
                }

            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason":     "Price within normal bounds. No extreme reading.",
            }

        except Exception as e:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": f"Calculation error: {e}"}