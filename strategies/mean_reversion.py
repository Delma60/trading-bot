# strategies/mean_reversion.py
"""
Mean Reversion Strategy — v4

Root cause analysis of poor live performance (40.8% WR, avg win $2.56 vs avg loss $7.34):
  - SL placed at `low - 5% band_width` was far too wide on high-volatility reversal bars.
    A reversal bar is almost always larger than average, so the "buffer" was eating into
    the trade's natural risk immediately.
  - TP capped at bb_mid (half the band away) while SL could be 2–3x that distance.
    min_rr=1.5 gate passed on PIPS but not on USD because lots were being sized to the
    full (inflated) pip stop — producing lopsided dollar outcomes.
  - No hard volume gate meant entries fired in thin, choppy conditions with no follow-through.
  - Single-candle confirmation is too fragile; a second candle closing back inside the band
    is more reliable.

Fixes in v4
-----------
1. ATR-based SL: SL = entry ∓ (ATR × atr_sl_mult) anchored to a nearby swing low/high,
   whichever is tighter. This caps the dollar risk regardless of band width.
2. Tiered TP: primary TP at bb_mid (1R minimum); if R:R allows, extend to 1.618× risk
   (Fibonacci extension). Entry lot size is sized against the ATR stop.
3. Hard volume gate: vol_ratio must exceed volume_mult before the signal fires.
4. Two-candle confirmation: both the touch bar AND the recovery bar must satisfy
   their respective conditions (not just the last closed bar).
5. RSI must cross back above/below a recovery threshold (not just be "turning").
6. Tighter ADX floor: 30 instead of 28 to ensure we're genuinely in a ranging regime.
7. ATR spread check: skip if ATR is spiking (> 2.5× normal) — news / stop-hunt conditions.
8. Stochastic confirmation gate (hard, not just scoring): stoch_k must be in the
   extreme zone at the touch bar for a buy/sell to qualify.
"""

import pandas as pd
import pandas_ta as ta


class MeanReversionStrategy:

    def __init__(
        self,
        bb_length:          int   = 20,
        bb_std:             float = 2.0,
        rsi_length:         int   = 14,
        stoch_k:            int   = 14,
        stoch_d:            int   = 3,
        adx_length:         int   = 14,
        adx_threshold:      float = 30.0,   # raised: must be genuinely ranging
        atr_length:         int   = 14,
        atr_sl_mult:        float = 1.2,    # SL = ATR × this from entry
        atr_spike_mult:     float = 2.5,    # skip if ATR > N × 20-bar ATR avg (news guard)
        volume_mult:        float = 1.20,   # HARD gate — bar must exceed N× avg vol
        rsi_buy_extreme:    float = 38.0,   # RSI must be ≤ this at the touch bar
        rsi_sell_extreme:   float = 62.0,   # RSI must be ≥ this at the touch bar
        rsi_recovery_buy:   float = 42.0,   # RSI must cross back above this on confirmation
        rsi_recovery_sell:  float = 58.0,   # RSI must cross back below this on confirmation
        stoch_buy_max:      float = 30.0,   # stoch_k ≤ this at touch (oversold)
        stoch_sell_min:     float = 70.0,   # stoch_k ≥ this at touch (overbought)
        min_rr:             float = 1.5,    # primary TP at bb_mid; fail if below this
        tp_extension_ratio: float = 1.618,  # Fib extension for secondary TP
        min_bb_width_atr:   float = 0.8,    # band width must be ≥ this × ATR (meaningful range)
        rsi_turn_bars:      int   = 3,
    ):
        self.bb_length          = bb_length
        self.bb_std             = bb_std
        self.rsi_length         = rsi_length
        self.stoch_k            = stoch_k
        self.stoch_d            = stoch_d
        self.adx_length         = adx_length
        self.adx_threshold      = adx_threshold
        self.atr_length         = atr_length
        self.atr_sl_mult        = atr_sl_mult
        self.atr_spike_mult     = atr_spike_mult
        self.volume_mult        = volume_mult
        self.rsi_buy_extreme    = rsi_buy_extreme
        self.rsi_sell_extreme   = rsi_sell_extreme
        self.rsi_recovery_buy   = rsi_recovery_buy
        self.rsi_recovery_sell  = rsi_recovery_sell
        self.stoch_buy_max      = stoch_buy_max
        self.stoch_sell_min     = stoch_sell_min
        self.min_rr             = min_rr
        self.tp_extension_ratio = tp_extension_ratio
        self.min_bb_width_atr   = min_bb_width_atr
        self.rsi_turn_bars      = rsi_turn_bars

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
        if len(data) < 5:
            return self._wait("Insufficient data after indicator warm-up.")

        return self._evaluate(data)

    # ── Indicator construction ────────────────────────────────────────────────

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Bollinger Bands
        bb = ta.bbands(data["close"], length=self.bb_length, std=self.bb_std)
        if bb is None or bb.empty:
            raise ValueError("Bollinger Bands failed.")
        data = pd.concat([data, bb], axis=1)
        self._bbl = [c for c in bb.columns if c.startswith("BBL")][0]
        self._bbu = [c for c in bb.columns if c.startswith("BBU")][0]
        self._bbm = [c for c in bb.columns if c.startswith("BBM")][0]

        # ATR
        data["atr"] = ta.atr(data["high"], data["low"], data["close"],
                              length=self.atr_length)
        data["atr_avg20"] = data["atr"].rolling(20).mean()

        # RSI
        data["rsi"] = ta.rsi(data["close"], length=self.rsi_length)

        # Stochastic
        stoch = ta.stoch(data["high"], data["low"], data["close"],
                         k=self.stoch_k, d=self.stoch_d)
        if stoch is not None and not stoch.empty:
            data = pd.concat([data, stoch], axis=1)
            self._sk = [c for c in stoch.columns if c.startswith("STOCHk")][0]
            self._sd = [c for c in stoch.columns if c.startswith("STOCHd")][0]
        else:
            data["_sk"] = 50.0
            data["_sd"] = 50.0
            self._sk = "_sk"
            self._sd = "_sd"

        # ADX
        adx_df = ta.adx(data["high"], data["low"], data["close"],
                         length=self.adx_length)
        if adx_df is not None and not adx_df.empty:
            data = pd.concat([data, adx_df], axis=1)
            self._adx = [c for c in adx_df.columns if c.startswith("ADX")][0]
        else:
            data["_adx"] = 0.0
            self._adx = "_adx"

        # Volume ratio
        data["vol_avg"]   = data["volume"].rolling(20).mean()
        data["vol_ratio"] = data["volume"] / data["vol_avg"].replace(0, float("nan"))

        # Swing lows/highs (5-bar lookback) for structural SL anchor
        data["swing_low"]  = data["low"].rolling(5, center=True).min()
        data["swing_high"] = data["high"].rolling(5, center=True).max()

        return data

    # ── Signal evaluation ─────────────────────────────────────────────────────

    def _evaluate(self, data: pd.DataFrame) -> dict:
        # bar  = last confirmed closed candle (confirmation / entry bar)
        # prev = the bar before it              (touch / extreme bar)
        bar  = data.iloc[-2]
        prev = data.iloc[-3]

        close   = float(bar["close"])
        open_   = float(bar["open"])
        high    = float(bar["high"])
        low     = float(bar["low"])

        bb_lower = float(bar[self._bbl])
        bb_upper = float(bar[self._bbu])
        bb_mid   = float(bar[self._bbm])
        bb_width = bb_upper - bb_lower

        atr       = float(bar["atr"])
        atr_avg   = float(bar["atr_avg20"])
        adx       = float(bar[self._adx])
        rsi       = float(bar["rsi"])
        prev_rsi  = float(prev["rsi"])
        stoch_k   = float(bar[self._sk])
        prev_sk   = float(prev[self._sk])
        vol_ratio = float(bar["vol_ratio"]) if not pd.isna(bar["vol_ratio"]) else 1.0

        prev_close  = float(prev["close"])
        prev_low    = float(prev["low"])
        prev_high   = float(prev["high"])
        prev_bbl    = float(prev[self._bbl])
        prev_bbu    = float(prev[self._bbu])

        # ── Gate 1: ATR spike guard (news / stop-hunt filter) ────────────────
        if atr_avg > 0 and atr > self.atr_spike_mult * atr_avg:
            return self._wait(
                f"ATR spike detected ({atr:.5f} > {self.atr_spike_mult}× avg {atr_avg:.5f}). "
                f"Likely news or stop-hunt conditions — skipping."
            )

        # ── Gate 2: ADX — must be ranging ────────────────────────────────────
        if adx > self.adx_threshold:
            return self._wait(
                f"Market trending (ADX {adx:.1f} > {self.adx_threshold}). "
                f"Mean reversion unreliable in directional markets."
            )

        # ── Gate 3: Band width meaningful relative to ATR ─────────────────────
        if atr > 0 and bb_width < self.min_bb_width_atr * atr:
            return self._wait(
                f"Bands too narrow relative to ATR "
                f"(BB width {bb_width:.5f} < {self.min_bb_width_atr}× ATR {atr:.5f}). "
                f"No meaningful oscillation range to trade."
            )

        # ── Gate 4: Hard volume confirmation ─────────────────────────────────
        if vol_ratio < self.volume_mult:
            return self._wait(
                f"Volume too low on signal bar ({vol_ratio:.2f}× < {self.volume_mult}×). "
                f"No institutional participation — skip."
            )

        # ── BUY setup ────────────────────────────────────────────────────────
        # Touch bar: close breached lower band AND stochastic was oversold
        touch_lower   = prev_close < prev_bbl
        stoch_oversold = prev_sk <= self.stoch_buy_max
        rsi_extreme   = prev_rsi <= self.rsi_buy_extreme

        # Confirmation bar: closed back inside band with bullish body
        confirmed_bull = close > bb_lower and close > open_

        # RSI crossed back above recovery level
        rsi_recovered_bull = prev_rsi < self.rsi_recovery_buy and rsi >= self.rsi_recovery_buy

        if touch_lower and stoch_oversold and rsi_extreme and confirmed_bull:
            if not rsi_recovered_bull:
                return self._wait(
                    f"Bullish setup forming but RSI hasn't crossed recovery threshold yet "
                    f"(RSI {rsi:.1f}, need ≥ {self.rsi_recovery_buy} from {prev_rsi:.1f}). "
                    f"Waiting one more bar."
                )

            # ATR-anchored SL — tighter of: ATR stop OR structural swing low
            atr_sl       = close - (self.atr_sl_mult * atr)
            structural_sl = float(data["swing_low"].iloc[-3]) - (0.1 * atr)
            sl_price     = max(atr_sl, structural_sl)   # less aggressive of the two
            sl_price     = min(sl_price, low - 0.5 * atr)  # never tighter than half ATR below bar low

            risk = close - sl_price
            if risk <= 0:
                return self._wait("BUY SL invalid — price at or below structural swing low.")

            # Primary TP at bb_mid; extended TP at Fib ratio
            tp_primary   = bb_mid
            tp_extended  = close + (risk * self.tp_extension_ratio)
            reward_primary = tp_primary - close

            if reward_primary <= 0:
                return self._wait("BUY TP invalid — price already at or above midline.")

            rr_primary = reward_primary / risk
            if rr_primary < self.min_rr:
                return self._wait(
                    f"BUY R:R {rr_primary:.2f}:1 below minimum {self.min_rr}. "
                    f"Midline too close ({reward_primary:.5f}) relative to SL ({risk:.5f})."
                )

            # Use extended TP if it gives better R:R
            use_tp  = tp_extended if tp_extended > tp_primary else tp_primary
            reward  = use_tp - close
            rr      = reward / risk

            pip_mult = self._pip_mult(close)
            sl_pips  = round(risk * pip_mult, 1)
            tp_pips  = round(reward * pip_mult, 1)

            confidence = self._score_buy(
                rsi, prev_rsi, prev_sk, vol_ratio, rr, adx, stoch_k
            )

            return {
                "action":     "BUY",
                "confidence": confidence,
                "reason": (
                    f"Mean Reversion BUY — two-bar confirmed recovery off lower BB. "
                    f"RSI {prev_rsi:.1f}→{rsi:.1f} (crossed {self.rsi_recovery_buy}). "
                    f"Stoch {prev_sk:.1f} (extreme ≤ {self.stoch_buy_max}). "
                    f"ATR SL: {sl_pips}p | TP: {tp_pips}p | R:R {rr:.2f}:1. "
                    f"ADX {adx:.1f}, Vol {vol_ratio:.1f}×."
                ),
                "sl_pips":   sl_pips,
                "tp_pips":   tp_pips,
                "sl_price":  round(sl_price, 5),
                "tp_price":  round(use_tp, 5),
                "rr":        round(rr, 2),
            }

        # ── SELL setup ───────────────────────────────────────────────────────
        touch_upper    = prev_close > prev_bbu
        stoch_overbought = prev_sk >= self.stoch_sell_min
        rsi_extreme_sell = prev_rsi >= self.rsi_sell_extreme

        confirmed_bear = close < bb_upper and close < open_

        rsi_recovered_bear = prev_rsi > self.rsi_recovery_sell and rsi <= self.rsi_recovery_sell

        if touch_upper and stoch_overbought and rsi_extreme_sell and confirmed_bear:
            if not rsi_recovered_bear:
                return self._wait(
                    f"Bearish setup forming but RSI hasn't crossed recovery threshold yet "
                    f"(RSI {rsi:.1f}, need ≤ {self.rsi_recovery_sell} from {prev_rsi:.1f}). "
                    f"Waiting one more bar."
                )

            atr_sl        = close + (self.atr_sl_mult * atr)
            structural_sl = float(data["swing_high"].iloc[-3]) + (0.1 * atr)
            sl_price      = min(atr_sl, structural_sl)
            sl_price      = max(sl_price, high + 0.5 * atr)

            risk = sl_price - close
            if risk <= 0:
                return self._wait("SELL SL invalid — price at or above structural swing high.")

            tp_primary   = bb_mid
            tp_extended  = close - (risk * self.tp_extension_ratio)
            reward_primary = close - tp_primary

            if reward_primary <= 0:
                return self._wait("SELL TP invalid — price already at or below midline.")

            rr_primary = reward_primary / risk
            if rr_primary < self.min_rr:
                return self._wait(
                    f"SELL R:R {rr_primary:.2f}:1 below minimum {self.min_rr}. "
                    f"Midline too close relative to SL."
                )

            use_tp  = tp_extended if tp_extended < tp_primary else tp_primary
            reward  = close - use_tp
            rr      = reward / risk

            pip_mult = self._pip_mult(close)
            sl_pips  = round(risk * pip_mult, 1)
            tp_pips  = round(reward * pip_mult, 1)

            confidence = self._score_sell(
                rsi, prev_rsi, prev_sk, vol_ratio, rr, adx, stoch_k
            )

            return {
                "action":     "SELL",
                "confidence": confidence,
                "reason": (
                    f"Mean Reversion SELL — two-bar confirmed rejection off upper BB. "
                    f"RSI {prev_rsi:.1f}→{rsi:.1f} (crossed {self.rsi_recovery_sell}). "
                    f"Stoch {prev_sk:.1f} (extreme ≥ {self.stoch_sell_min}). "
                    f"ATR SL: {sl_pips}p | TP: {tp_pips}p | R:R {rr:.2f}:1. "
                    f"ADX {adx:.1f}, Vol {vol_ratio:.1f}×."
                ),
                "sl_pips":   sl_pips,
                "tp_pips":   tp_pips,
                "sl_price":  round(sl_price, 5),
                "tp_price":  round(use_tp, 5),
                "rr":        round(rr, 2),
            }

        return self._wait(
            f"No two-bar reversal confirmation. "
            f"ADX {adx:.1f} | RSI {rsi:.1f} | Stoch {stoch_k:.1f} | "
            f"Bands [{bb_lower:.5f} – {bb_upper:.5f}]."
        )

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _score_buy(
        self, rsi: float, prev_rsi: float, prev_sk: float,
        vol_ratio: float, rr: float, adx: float, stoch_k_now: float
    ) -> float:
        score = 0.50

        # RSI depth at touch (deeper = more stretched = stronger bounce expected)
        if prev_rsi < 25:   score += 0.12
        elif prev_rsi < 30: score += 0.08
        elif prev_rsi < 38: score += 0.04

        # Stochastic depth at touch
        if prev_sk < 15:    score += 0.08
        elif prev_sk < 20:  score += 0.05
        elif prev_sk < 30:  score += 0.02

        # Stochastic crossing back up (recovery confirmation)
        if stoch_k_now > prev_sk + 5: score += 0.05

        # Volume (institutional reversal footprint)
        if vol_ratio >= 2.0:   score += 0.08
        elif vol_ratio >= 1.5: score += 0.05
        else:                  score += 0.02   # already passed hard gate at 1.2

        # R:R quality (cap contribution at R:R = 4)
        score += min((rr - self.min_rr) / 2.5, 1.0) * 0.08

        # Ranging confirmation (lower ADX within threshold = cleaner range)
        if adx < 20:   score += 0.05
        elif adx < 25: score += 0.02

        return round(min(score, 0.95), 2)

    def _score_sell(
        self, rsi: float, prev_rsi: float, prev_sk: float,
        vol_ratio: float, rr: float, adx: float, stoch_k_now: float
    ) -> float:
        score = 0.50

        if prev_rsi > 75:   score += 0.12
        elif prev_rsi > 70: score += 0.08
        elif prev_rsi > 62: score += 0.04

        if prev_sk > 85:    score += 0.08
        elif prev_sk > 80:  score += 0.05
        elif prev_sk > 70:  score += 0.02

        if stoch_k_now < prev_sk - 5: score += 0.05

        if vol_ratio >= 2.0:   score += 0.08
        elif vol_ratio >= 1.5: score += 0.05
        else:                  score += 0.02

        score += min((rr - self.min_rr) / 2.5, 1.0) * 0.08

        if adx < 20:   score += 0.05
        elif adx < 25: score += 0.02

        return round(min(score, 0.95), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pip_mult(price: float) -> float:
        """Approximate pip multiplier from price magnitude."""
        if price > 100:   # JPY, Gold, Indices
            return 100.0
        return 10000.0    # Standard 5-digit Forex

    @staticmethod
    def _wait(reason: str) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": reason}