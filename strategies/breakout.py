# strategies/breakout.py
import pandas as pd
import pandas_ta as ta


class BreakoutStrategy:
    """
    Identifies high-conviction price breakouts through historical support/resistance.

    Improvements over v1
    --------------------
    * Volume confirmation  — breakout candle must have above-average volume.
    * ATR buffer           — close must exceed S/R by a minimum margin to
                             filter noise and micro-breakouts.
    * Bollinger squeeze    — detects compression (low BB width) before the
                             move, increasing confidence when a squeeze is present.
    * RSI alignment        — bullish breakout needs RSI > 50; bearish < 50.
    * Candle close quality — close must be in the top/bottom 25 % of the bar
                             to prove conviction.
    * Graduated confidence — scales continuously with distance, volume surge,
                             ATR buffer, and RSI momentum rather than a hard jump.
    """

    def __init__(
        self,
        lookback_window:   int   = 20,
        atr_length:        int   = 14,
        volume_multiplier: float = 1.3,    # volume must exceed N × 20-bar avg
        atr_buffer_pct:    float = 0.15,   # close must be buffer × ATR past S/R
        bb_squeeze_pct:    float = 0.02,   # BB width / price < this → squeeze
    ):
        self.lookback_window   = lookback_window
        self.atr_length        = atr_length
        self.volume_multiplier = volume_multiplier
        self.atr_buffer_pct    = atr_buffer_pct
        self.bb_squeeze_pct    = bb_squeeze_pct

    def analyze(self, df: pd.DataFrame) -> dict:
        min_bars = self.lookback_window + self.atr_length + 2
        if df is None or len(df) < min_bars:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": f"Not enough data (need {min_bars} bars)."}

        data = df.copy()

        # ── Indicators ──────────────────────────────────────────────────────
        # Support & resistance from the PREVIOUS window (shift avoids lookahead)
        data["resistance"] = data["high"].rolling(self.lookback_window).max().shift(1)
        data["support"]    = data["low"].rolling(self.lookback_window).min().shift(1)

        data["atr"]        = ta.atr(data["high"], data["low"], data["close"],
                                    length=self.atr_length)
        data["rsi"]        = ta.rsi(data["close"], length=14)

        # Volume surge: is this candle's volume above the rolling average?
        data["vol_avg"]    = data["volume"].rolling(20).mean()
        data["vol_ratio"]  = data["volume"] / data["vol_avg"].replace(0, float("nan"))

        # Bollinger squeeze: narrow bands signal coiled energy
        bb = ta.bbands(data["close"], length=20, std=2.0)
        if bb is not None and not bb.empty:
            bbl = [c for c in bb.columns if c.startswith("BBL")][0]
            bbu = [c for c in bb.columns if c.startswith("BBU")][0]
            bbm = [c for c in bb.columns if c.startswith("BBM")][0]
            data["bb_width"] = (bb[bbu] - bb[bbl]) / bb[bbm].replace(0, float("nan"))
        else:
            data["bb_width"] = float("nan")

        data.dropna(inplace=True)
        if data.empty:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": "Insufficient data after indicator warm-up."}

        # ── Snapshot (use the last *confirmed* closed candle) ────────────────
        bar = data.iloc[-2]
        close      = bar["close"]
        open_      = bar["open"]
        high       = bar["high"]
        low        = bar["low"]
        resistance = bar["resistance"]
        support    = bar["support"]
        atr        = bar["atr"]
        rsi        = bar["rsi"]
        vol_ratio  = bar["vol_ratio"]
        bb_width   = bar["bb_width"]

        # ── Shared pre-conditions ────────────────────────────────────────────
        body_size    = abs(close - open_)
        bar_range    = high - low + 1e-9
        atr_buffer   = self.atr_buffer_pct * atr

        # Candle close quality: close in top 25 % of bar (bull) / bottom 25 % (bear)
        close_quality_bull = (close - low)  / bar_range   # 0 = bottom, 1 = top
        close_quality_bear = (high - close) / bar_range

        volume_confirmed = vol_ratio >= self.volume_multiplier
        strong_body      = body_size > (atr * 0.4)
        squeeze_present  = (not pd.isna(bb_width)) and (bb_width < self.bb_squeeze_pct)

        # ── BUY ─────────────────────────────────────────────────────────────
        if (
            close > resistance + atr_buffer   # meaningful break, not just touch
            and strong_body
            and close_quality_bull > 0.75      # closed near candle top
            and rsi > 50                       # momentum aligned
        ):
            if not volume_confirmed:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Bullish breakout signal, but volume too low "
                                  f"(ratio: {vol_ratio:.2f} < {self.volume_multiplier})."}

            distance_score = min((close - resistance) / atr, 1.0)    # 0–1
            volume_score   = min((vol_ratio - 1.0) / 2.0, 1.0)       # 0–1
            rsi_score      = min((rsi - 50) / 50.0, 1.0)             # 0–1
            squeeze_bonus  = 0.05 if squeeze_present else 0.0

            confidence = (
                0.60
                + 0.15 * distance_score
                + 0.10 * volume_score
                + 0.08 * rsi_score
                + squeeze_bonus
            )
            confidence = round(min(confidence, 0.97), 2)

            return {
                "action":     "BUY",
                "confidence": confidence,
                "reason": (
                    f"Bullish Breakout above {self.lookback_window}-bar resistance "
                    f"({resistance:.5f}). Close: {close:.5f}, ATR buffer met, "
                    f"Vol ratio: {vol_ratio:.2f}x, RSI: {rsi:.1f}"
                    + (" [squeeze release]" if squeeze_present else "")
                ),
            }

        # ── SELL ─────────────────────────────────────────────────────────────
        if (
            close < support - atr_buffer
            and strong_body
            and close_quality_bear > 0.75      # closed near candle bottom
            and rsi < 50
        ):
            if not volume_confirmed:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Bearish breakout signal, but volume too low "
                                  f"(ratio: {vol_ratio:.2f} < {self.volume_multiplier})."}

            distance_score = min((support - close) / atr, 1.0)
            volume_score   = min((vol_ratio - 1.0) / 2.0, 1.0)
            rsi_score      = min((50 - rsi) / 50.0, 1.0)
            squeeze_bonus  = 0.05 if squeeze_present else 0.0

            confidence = (
                0.60
                + 0.15 * distance_score
                + 0.10 * volume_score
                + 0.08 * rsi_score
                + squeeze_bonus
            )
            confidence = round(min(confidence, 0.97), 2)

            return {
                "action":     "SELL",
                "confidence": confidence,
                "reason": (
                    f"Bearish Breakout below {self.lookback_window}-bar support "
                    f"({support:.5f}). Close: {close:.5f}, ATR buffer met, "
                    f"Vol ratio: {vol_ratio:.2f}x, RSI: {rsi:.1f}"
                    + (" [squeeze release]" if squeeze_present else "")
                ),
            }

        # ── WAIT ─────────────────────────────────────────────────────────────
        squeeze_note = " (Squeeze building — watch for breakout.)" if squeeze_present else ""
        return {
            "action":     "WAIT",
            "confidence": 0.0,
            "reason":     f"Price consolidating inside range "
                          f"[{support:.5f} – {resistance:.5f}].{squeeze_note}",
        }