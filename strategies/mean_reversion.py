# strategies/mean_reversion.py
"""
Mean Reversion Strategy — Refactored

Root cause of prior underperformance:
  - Entered on first band touch instead of confirmed reversal
  - No R:R gate: wins averaged $0.05 vs losses $0.19
  - SL/TP not returned in signal, so execution used generic fixed pips
  - RSI checked for level, not direction — stale extremes caused bad entries

Fixes applied:
  1. Confirmation candle: price must close BACK inside the band before entry
  2. RSI must be turning away from extreme, not just sitting there
  3. Natural SL/TP derived from the band geometry and returned in signal
  4. Hard R:R gate: skips trade if potential R < 1.5 : 1
  5. Minimum retracement to midline: ensures enough room to profit
  6. Squeeze filter: skips when bands are too narrow (no room to profit)
  7. Graduated confidence tied to actual R:R ratio achieved
"""

import pandas as pd
import pandas_ta as ta


class MeanReversionStrategy:

    def __init__(
        self,
        bb_length:        int   = 20,
        bb_std:           float = 2.0,
        rsi_length:       int   = 14,
        stoch_k:          int   = 14,
        stoch_d:          int   = 3,
        adx_length:       int   = 14,
        adx_threshold:    float = 28.0,   # above → trending, skip
        volume_mult:      float = 1.15,   # volume spike confirmation
        min_rr:           float = 1.5,    # hard R:R floor before signalling
        min_midline_dist: float = 0.003,  # min % distance price→midline (room to profit)
        min_bb_width:     float = 0.005,  # min % BB width — skip squeezed markets
        rsi_turn_bars:    int   = 3,      # lookback bars to detect RSI turning
    ):
        self.bb_length        = bb_length
        self.bb_std           = bb_std
        self.rsi_length       = rsi_length
        self.stoch_k          = stoch_k
        self.stoch_d          = stoch_d
        self.adx_length       = adx_length
        self.adx_threshold    = adx_threshold
        self.volume_mult      = volume_mult
        self.min_rr           = min_rr
        self.min_midline_dist = min_midline_dist
        self.min_bb_width     = min_bb_width
        self.rsi_turn_bars    = rsi_turn_bars

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> dict:
        min_bars = max(200, self.bb_length + 50, self.rsi_length + self.rsi_turn_bars + 5)
        if df is None or len(df) < min_bars:
            return self._wait(f"Need at least {min_bars} bars.")

        data = df.copy()

        try:
            data = self._add_indicators(data)
        except Exception as e:
            return self._wait(f"Indicator error: {e}")

        data.dropna(inplace=True)
        if len(data) < 4:
            return self._wait("Insufficient data after indicator warm-up.")

        return self._evaluate(data)

    # ── Indicator construction ────────────────────────────────────────────────

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Bollinger Bands
        bb = ta.bbands(data["close"], length=self.bb_length, std=self.bb_std)
        if bb is None or bb.empty:
            raise ValueError("Bollinger Bands failed.")
        data = pd.concat([data, bb], axis=1)
        self._bb_lower = [c for c in bb.columns if c.startswith("BBL")][0]
        self._bb_upper = [c for c in bb.columns if c.startswith("BBU")][0]
        self._bb_mid   = [c for c in bb.columns if c.startswith("BBM")][0]

        # RSI
        data["rsi"] = ta.rsi(data["close"], length=self.rsi_length)

        # Stochastic
        stoch = ta.stoch(data["high"], data["low"], data["close"],
                         k=self.stoch_k, d=self.stoch_d)
        if stoch is not None and not stoch.empty:
            data = pd.concat([data, stoch], axis=1)
            self._stoch_k = [c for c in stoch.columns if c.startswith("STOCHk")][0]
            self._stoch_d = [c for c in stoch.columns if c.startswith("STOCHd")][0]
        else:
            data["_stoch_k"] = 50.0
            data["_stoch_d"] = 50.0
            self._stoch_k = "_stoch_k"
            self._stoch_d = "_stoch_d"

        # ADX
        adx_df = ta.adx(data["high"], data["low"], data["close"], length=self.adx_length)
        if adx_df is not None and not adx_df.empty:
            data = pd.concat([data, adx_df], axis=1)
            self._adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
        else:
            data["_adx"] = 0.0
            self._adx_col = "_adx"

        # Volume ratio
        data["vol_avg"]   = data["volume"].rolling(20).mean()
        data["vol_ratio"] = data["volume"] / data["vol_avg"].replace(0, float("nan"))

        return data

    # ── Signal evaluation ─────────────────────────────────────────────────────

    def _evaluate(self, data: pd.DataFrame) -> dict:
        # Index -1 = live (potentially open) candle
        # Index -2 = last confirmed closed candle  (signal candle)
        # Index -3 = the candle before that        (touch candle for confirmation pattern)
        bar     = data.iloc[-2]   # signal bar — must close back inside the band
        prev    = data.iloc[-3]   # prior bar — the one that touched/breached the band

        close   = float(bar["close"])
        open_   = float(bar["open"])
        low     = float(bar["low"])
        high    = float(bar["high"])

        bb_lower  = float(bar[self._bb_lower])
        bb_upper  = float(bar[self._bb_upper])
        bb_mid    = float(bar[self._bb_mid])
        bb_width  = (bb_upper - bb_lower) / (bb_mid + 1e-9)

        adx       = float(bar[self._adx_col])
        rsi       = float(bar["rsi"])
        stoch_k   = float(bar[self._stoch_k])
        stoch_d   = float(bar[self._stoch_d])
        vol_ratio = float(bar["vol_ratio"]) if not pd.isna(bar["vol_ratio"]) else 1.0

        prev_close  = float(prev["close"])
        prev_lower  = float(prev[self._bb_lower])
        prev_upper  = float(prev[self._bb_upper])

        # ── Gate 1: ADX — must be ranging ────────────────────────────────────
        if adx > self.adx_threshold:
            return self._wait(
                f"Market is trending (ADX {adx:.1f} > {self.adx_threshold}). "
                f"Mean reversion is unreliable in trends."
            )

        # ── Gate 2: Band width — skip squeezed markets ────────────────────────
        # Narrow bands mean little room between entry and target; not worth the risk.
        if bb_width < self.min_bb_width:
            return self._wait(
                f"Bands too narrow (width {bb_width:.4f}). "
                f"Insufficient room to profit; waiting for expansion."
            )

        # ── Gate 3: RSI momentum direction ───────────────────────────────────
        # RSI must be turning back from extreme, not stalling at it.
        rsi_series = data["rsi"].iloc[-(self.rsi_turn_bars + 2):-1]
        rsi_peak   = float(rsi_series.max())
        rsi_trough = float(rsi_series.min())
        rsi_turning_up   = rsi > rsi_trough + 2.0   # RSI bouncing from its recent low
        rsi_turning_down = rsi < rsi_peak  - 2.0    # RSI falling from its recent high

        # ── BUY path ─────────────────────────────────────────────────────────
        # Confirmation pattern: previous bar closed BELOW lower band,
        # current bar closes back INSIDE (above lower band) with a bullish body.
        prev_breached_lower = prev_close < prev_lower
        current_reclaimed   = close > bb_lower        # back inside
        bullish_body        = close > open_            # green candle

        if prev_breached_lower and current_reclaimed and bullish_body:
            if not rsi_turning_up:
                return self._wait(
                    f"Bullish reversal candle present but RSI not yet turning "
                    f"(RSI {rsi:.1f}, trough {rsi_trough:.1f}). Waiting."
                )

            # Natural targets: SL below the signal bar low, TP at midline
            sl_price = low - (bb_upper - bb_lower) * 0.05   # 5% of band width below the low
            tp_price = bb_mid

            # R:R gate
            risk   = close - sl_price
            reward = tp_price - close
            if risk <= 0 or reward <= 0:
                return self._wait("BUY targets invalid (price at or above midline).")

            rr = reward / risk
            if rr < self.min_rr:
                return self._wait(
                    f"R:R {rr:.2f} below minimum {self.min_rr}. "
                    f"Not enough room to midline to justify the risk."
                )

            # Distance to midline as % of price — ensures meaningful profit potential
            midline_dist = reward / close
            if midline_dist < self.min_midline_dist:
                return self._wait(
                    f"Midline too close ({midline_dist:.3%}). "
                    f"Not enough upside to the mean."
                )

            # Convert SL/TP to pips for the execution layer
            pip_mult   = self._pip_multiplier(close)
            sl_pips    = round(risk * pip_mult, 1)
            tp_pips    = round(reward * pip_mult, 1)

            confidence = self._score_buy(rsi, stoch_k, stoch_d, vol_ratio, rr, adx)
            return {
                "action":     "BUY",
                "confidence": confidence,
                "reason":     (
                    f"Mean Reversion BUY — confirmed reversal off lower band. "
                    f"RSI {rsi:.1f} turning up. "
                    f"R:R {rr:.2f}:1 | SL {sl_pips}p → TP {tp_pips}p (midline). "
                    f"ADX {adx:.1f}, Vol {vol_ratio:.1f}x."
                ),
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "sl_price": round(sl_price, 5),
                "tp_price": round(tp_price, 5),
                "rr":       round(rr, 2),
            }

        # ── SELL path ─────────────────────────────────────────────────────────
        prev_breached_upper = prev_close > prev_upper
        current_reclaimed   = close < bb_upper
        bearish_body        = close < open_

        if prev_breached_upper and current_reclaimed and bearish_body:
            if not rsi_turning_down:
                return self._wait(
                    f"Bearish reversal candle present but RSI not yet turning "
                    f"(RSI {rsi:.1f}, peak {rsi_peak:.1f}). Waiting."
                )

            sl_price = high + (bb_upper - bb_lower) * 0.05
            tp_price = bb_mid

            risk   = sl_price - close
            reward = close - tp_price
            if risk <= 0 or reward <= 0:
                return self._wait("SELL targets invalid (price at or below midline).")

            rr = reward / risk
            if rr < self.min_rr:
                return self._wait(
                    f"R:R {rr:.2f} below minimum {self.min_rr}. "
                    f"Not enough room to midline to justify the risk."
                )

            midline_dist = reward / close
            if midline_dist < self.min_midline_dist:
                return self._wait(
                    f"Midline too close ({midline_dist:.3%}). "
                    f"Not enough downside to the mean."
                )

            pip_mult   = self._pip_multiplier(close)
            sl_pips    = round(risk * pip_mult, 1)
            tp_pips    = round(reward * pip_mult, 1)

            confidence = self._score_sell(rsi, stoch_k, stoch_d, vol_ratio, rr, adx)
            return {
                "action":     "SELL",
                "confidence": confidence,
                "reason":     (
                    f"Mean Reversion SELL — confirmed reversal off upper band. "
                    f"RSI {rsi:.1f} turning down. "
                    f"R:R {rr:.2f}:1 | SL {sl_pips}p → TP {tp_pips}p (midline). "
                    f"ADX {adx:.1f}, Vol {vol_ratio:.1f}x."
                ),
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "sl_price": round(sl_price, 5),
                "tp_price": round(tp_price, 5),
                "rr":       round(rr, 2),
            }

        return self._wait(
            f"No confirmed reversal candle. ADX {adx:.1f}, RSI {rsi:.1f}. "
            f"Price within bands [{bb_lower:.5f} – {bb_upper:.5f}]."
        )

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _score_buy(self, rsi, stoch_k, stoch_d, vol_ratio, rr, adx) -> float:
        score = 0.50

        # RSI depth (deeper = stronger mean reversion pressure)
        if rsi < 25:
            score += 0.12
        elif rsi < 30:
            score += 0.08
        elif rsi < 40:
            score += 0.04

        # Stochastic oversold confirmation
        if stoch_k < 20 and stoch_d < 20:
            score += 0.08
        elif stoch_k < 30:
            score += 0.04

        # Volume: institutional footprint at the reversal
        if vol_ratio >= 1.5:
            score += 0.07
        elif vol_ratio >= self.volume_mult:
            score += 0.04

        # R:R quality (capped at R:R = 3)
        score += min((rr - self.min_rr) / (3.0 - self.min_rr), 1.0) * 0.08

        # Ranging confirmation (lower ADX = more ranging = better for mean rev)
        if adx < 20:
            score += 0.05

        return round(min(score, 0.95), 2)

    def _score_sell(self, rsi, stoch_k, stoch_d, vol_ratio, rr, adx) -> float:
        score = 0.50

        if rsi > 75:
            score += 0.12
        elif rsi > 70:
            score += 0.08
        elif rsi > 60:
            score += 0.04

        if stoch_k > 80 and stoch_d > 80:
            score += 0.08
        elif stoch_k > 70:
            score += 0.04

        if vol_ratio >= 1.5:
            score += 0.07
        elif vol_ratio >= self.volume_mult:
            score += 0.04

        score += min((rr - self.min_rr) / (3.0 - self.min_rr), 1.0) * 0.08

        if adx < 20:
            score += 0.05

        return round(min(score, 0.95), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pip_multiplier(price: float) -> float:
        """Rough pip multiplier based on price magnitude."""
        if price > 100:    # JPY pairs, Gold, indices
            return 100.0
        return 10000.0     # Standard forex (5-digit)

    @staticmethod
    def _wait(reason: str) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": reason}