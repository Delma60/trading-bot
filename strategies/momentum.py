# strategies/momentum.py
"""
Momentum Strategy — v3 (Sophisticated)

Architecture
------------
Three-phase signal pipeline:

  Phase 1 — Regime Filter
    Only trade when the market is actually trending.
    Uses ADX + Supertrend to confirm directional conviction before
    any entry logic runs. This alone eliminates the majority of
    false signals that occur in ranging/choppy conditions.

  Phase 2 — Entry Timing
    Waits for a pullback within the trend, then confirms resumption.
    Uses MACD histogram acceleration + RSI divergence detection to
    distinguish fresh momentum from exhaustion. Adds Stochastic
    crossover as a precision timing trigger to avoid chasing.

  Phase 3 — Dynamic SL/TP & Quality Scoring
    SL placed at the structural swing low/high using ATR buffer.
    TP scales dynamically with confluence strength — more confirming
    factors = wider TP target. Graduated 7-factor confidence scoring.

Key upgrades over v1
---------------------
  - Supertrend replaces bare EMA (fewer whipsaws, cleaner direction)
  - DI spread confirms directionality beyond raw ADX
  - Pullback detection: avoids chasing extended moves
  - RSI hidden divergence: flags exhausted momentum before reversal
  - Stochastic crossover: precise timing within the pullback
  - Multi-bar MACD histogram acceleration check
  - Volume confirmation on signal bar
  - Dynamic TP scaling based on confluence count (2.5× → 4.0× ATR)
  - Structural SL at recent pivot + ATR buffer
  - Returns sl_pips, tp_pips, sl_price, tp_price for execution layer
"""

import pandas as pd
import pandas_ta as ta


class MomentumStrategy:

    def __init__(
        self,
        # MACD
        macd_fast:          int   = 12,
        macd_slow:          int   = 26,
        macd_signal:        int   = 9,
        # Trend filters
        ema_length:         int   = 20,
        adx_length:         int   = 14,
        adx_min:            float = 25.0,   # below = ranging, skip
        supertrend_period:  int   = 10,
        supertrend_mult:    float = 3.0,
        # RSI
        rsi_length:         int   = 14,
        rsi_bull_min:       float = 55.0,   # RSI must clear this on BUY
        rsi_bear_max:       float = 45.0,   # RSI must be below this on SELL
        # Stochastic timing
        stoch_k:            int   = 5,
        stoch_d:            int   = 3,
        # Volume
        volume_mult:        float = 1.15,   # signal bar must exceed N× avg vol
        # Pullback detection
        pullback_bars:      int   = 6,      # lookback window for dip/rip detection
        pullback_rsi_max:   float = 55.0,   # RSI must have dipped below here (bull)
        pullback_rsi_min:   float = 45.0,   # RSI must have spiked above here (bear)
        # Histogram acceleration
        hist_accel_bars:    int   = 2,      # consecutive bars histogram must grow
        # Risk management
        atr_sl_mult:        float = 1.5,    # SL = swing_extreme - ATR × mult
        atr_tp_base_mult:   float = 2.5,    # TP base multiplier
        atr_tp_max_mult:    float = 4.0,    # TP ceiling when confluence is strong
        min_rr:             float = 1.5,    # hard R:R floor
    ):
        self.macd_fast         = macd_fast
        self.macd_slow         = macd_slow
        self.macd_signal       = macd_signal
        self.ema_length        = ema_length
        self.adx_length        = adx_length
        self.adx_min           = adx_min
        self.supertrend_period = supertrend_period
        self.supertrend_mult   = supertrend_mult
        self.rsi_length        = rsi_length
        self.rsi_bull_min      = rsi_bull_min
        self.rsi_bear_max      = rsi_bear_max
        self.stoch_k           = stoch_k
        self.stoch_d           = stoch_d
        self.volume_mult       = volume_mult
        self.pullback_bars     = pullback_bars
        self.pullback_rsi_max  = pullback_rsi_max
        self.pullback_rsi_min  = pullback_rsi_min
        self.hist_accel_bars   = hist_accel_bars
        self.atr_sl_mult       = atr_sl_mult
        self.atr_tp_base_mult  = atr_tp_base_mult
        self.atr_tp_max_mult   = atr_tp_max_mult
        self.min_rr            = min_rr

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> dict:
        min_bars = self.macd_slow + self.macd_signal + self.adx_length + 30
        if df is None or len(df) < min_bars:
            return self._wait(f"Need at least {min_bars} bars.")

        data = df.copy()

        try:
            data = self._add_indicators(data)
        except Exception as exc:
            return self._wait(f"Indicator error: {exc}")

        data.dropna(inplace=True)
        if len(data) < self.pullback_bars + 5:
            return self._wait("Insufficient data after indicator warm-up.")

        return self._evaluate(data)

    # ── Indicator construction ────────────────────────────────────────────────

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # MACD
        macd_df = ta.macd(
            data["close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        if macd_df is None or macd_df.empty:
            raise ValueError("MACD calculation failed.")
        data = pd.concat([data, macd_df], axis=1)
        self._macd_line = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        self._macd_hist = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        self._macd_sig  = f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

        # EMA
        data["ema"] = ta.ema(data["close"], length=self.ema_length)

        # RSI
        data["rsi"] = ta.rsi(data["close"], length=self.rsi_length)

        # ADX + Directional Indicators
        adx_df = ta.adx(data["high"], data["low"], data["close"], length=self.adx_length)
        if adx_df is not None and not adx_df.empty:
            data = pd.concat([data, adx_df], axis=1)
            self._adx_col      = f"ADX_{self.adx_length}"
            self._plus_di_col  = f"DMP_{self.adx_length}"
            self._minus_di_col = f"DMN_{self.adx_length}"
        else:
            data["_adx"] = 0.0
            data["_pdi"] = 0.0
            data["_mdi"] = 0.0
            self._adx_col      = "_adx"
            self._plus_di_col  = "_pdi"
            self._minus_di_col = "_mdi"

        # ATR
        data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=14)

        # Supertrend
        st_df = ta.supertrend(
            data["high"], data["low"], data["close"],
            length=self.supertrend_period,
            multiplier=self.supertrend_mult,
        )
        if st_df is not None and not st_df.empty:
            data = pd.concat([data, st_df], axis=1)
            # Direction column: +1 = bullish, -1 = bearish
            dir_cols = [c for c in st_df.columns if "SUPERTd" in c]
            self._st_dir_col = dir_cols[0] if dir_cols else "_st_dir"
        else:
            data["_st_dir"] = 1
            self._st_dir_col = "_st_dir"

        # Stochastic
        stoch_df = ta.stoch(
            data["high"], data["low"], data["close"],
            k=self.stoch_k, d=self.stoch_d, smooth_k=3,
        )
        if stoch_df is not None and not stoch_df.empty:
            data = pd.concat([data, stoch_df], axis=1)
            self._stoch_k_col = f"STOCHk_{self.stoch_k}_{self.stoch_d}_3"
            self._stoch_d_col = f"STOCHd_{self.stoch_k}_{self.stoch_d}_3"
        else:
            data["_sk"] = 50.0
            data["_sd"] = 50.0
            self._stoch_k_col = "_sk"
            self._stoch_d_col = "_sd"

        # Volume ratio
        data["vol_avg"]   = data["volume"].rolling(20).mean()
        data["vol_ratio"] = data["volume"] / data["vol_avg"].replace(0, float("nan"))

        return data

    # ── Three-phase evaluation ────────────────────────────────────────────────

    def _evaluate(self, data: pd.DataFrame) -> dict:
        bar  = data.iloc[-2]   # last confirmed closed candle
        prev = data.iloc[-3]

        close     = float(bar["close"])
        ema       = float(bar["ema"])
        rsi       = float(bar["rsi"])
        adx       = float(bar[self._adx_col])
        plus_di   = float(bar[self._plus_di_col])
        minus_di  = float(bar[self._minus_di_col])
        atr       = float(bar["atr"])
        st_dir    = int(bar[self._st_dir_col])
        stoch_k   = float(bar[self._stoch_k_col])
        stoch_d   = float(bar[self._stoch_d_col])
        prev_sk   = float(prev[self._stoch_k_col])
        prev_sd   = float(prev[self._stoch_d_col])
        vol_ratio = float(bar["vol_ratio"]) if not pd.isna(bar["vol_ratio"]) else 1.0
        macd_line = float(bar[self._macd_line])
        macd_hist = float(bar[self._macd_hist])
        prev_hist = float(prev[self._macd_hist])

        # ── Phase 1: Regime gate ──────────────────────────────────────────────
        if adx < self.adx_min:
            return self._wait(
                f"Regime: ranging (ADX {adx:.1f} < {self.adx_min}). "
                f"Momentum is unreliable without directional conviction."
            )

        # ── Phase 2: Histogram acceleration ──────────────────────────────────
        hist_window = data[self._macd_hist].iloc[-(self.hist_accel_bars + 2):-1]
        hist_accel_up   = all(
            hist_window.iloc[i] > hist_window.iloc[i - 1]
            for i in range(1, len(hist_window))
        )
        hist_accel_down = all(
            hist_window.iloc[i] < hist_window.iloc[i - 1]
            for i in range(1, len(hist_window))
        )

        # ── Phase 2: Pullback detection ───────────────────────────────────────
        rsi_window = data["rsi"].iloc[-(self.pullback_bars + 2):-1]
        had_bull_pullback = float(rsi_window.min()) <= self.pullback_rsi_max
        had_bear_pullback = float(rsi_window.max()) >= self.pullback_rsi_min

        # ── Phase 2: Hidden divergence detection ─────────────────────────────
        # Hidden bearish: price making higher highs, RSI not confirming → exhaustion
        # Hidden bullish: price making lower lows, RSI not confirming → strength hidden
        price_window = data["close"].iloc[-(self.pullback_bars + 2):-1]
        rsi_is_hh    = float(rsi_window.iloc[-1])   > float(rsi_window.iloc[0])
        rsi_is_ll    = float(rsi_window.iloc[-1])   < float(rsi_window.iloc[0])
        price_is_hh  = float(price_window.iloc[-1]) > float(price_window.iloc[0])
        price_is_ll  = float(price_window.iloc[-1]) < float(price_window.iloc[0])

        hidden_bear_div = price_is_hh and not rsi_is_hh   # bull momentum fading
        hidden_bull_div = price_is_ll and not rsi_is_ll   # bear momentum fading

        # ── Phase 2: Stochastic crossover timing ──────────────────────────────
        stoch_bull_cross = (prev_sk <= prev_sd) and (stoch_k > stoch_d)
        stoch_bear_cross = (prev_sk >= prev_sd) and (stoch_k < stoch_d)

        # ── Volume gate ───────────────────────────────────────────────────────
        if vol_ratio < self.volume_mult:
            return self._wait(
                f"Volume ({vol_ratio:.2f}×) below required {self.volume_mult}×. "
                f"Momentum without volume confirmation has poor follow-through."
            )

        # ─────────────────────────────────────────────────────────────────────
        # BUY path
        # ─────────────────────────────────────────────────────────────────────
        bull_regime = (st_dir == 1) and (close > ema) and (plus_di > minus_di)
        bull_macd   = (macd_line > 0) and (macd_hist > prev_hist)
        bull_rsi    = rsi >= self.rsi_bull_min
        bull_timing = stoch_bull_cross and stoch_k < 80

        if bull_regime:
            if hidden_bear_div:
                return self._wait(
                    f"Hidden bearish divergence — price at higher high but RSI "
                    f"not confirming ({rsi:.1f}). Momentum likely fading into resistance."
                )
            if not bull_macd:
                return self._wait(
                    f"Bull regime confirmed (ADX {adx:.1f}, Supertrend bullish, "
                    f"+DI > −DI) but MACD histogram not accelerating yet."
                )
            if not bull_rsi:
                return self._wait(
                    f"Bull regime present but RSI {rsi:.1f} < {self.rsi_bull_min}. "
                    f"Waiting for momentum to build."
                )
            if not had_bull_pullback:
                return self._wait(
                    f"Bull trend valid but no pullback in last {self.pullback_bars} bars "
                    f"(RSI min {float(rsi_window.min()):.1f} vs threshold {self.pullback_rsi_max}). "
                    f"Avoiding chase entries — waiting for dip."
                )

            # Phase 3: Structural SL/TP
            swing_low = float(data["low"].iloc[-(self.pullback_bars + 2):-1].min())
            sl_price  = swing_low - (self.atr_sl_mult * atr)
            risk      = close - sl_price
            if risk <= 0:
                return self._wait("BUY SL calculation invalid (swing low above close).")

            confluence = self._count_bull_confluences(
                hist_accel_up, bull_timing, had_bull_pullback, hidden_bull_div, vol_ratio
            )
            tp_mult  = self.atr_tp_base_mult + (confluence / 5.0) * (self.atr_tp_max_mult - self.atr_tp_base_mult)
            tp_price = close + (tp_mult * atr)
            reward   = tp_price - close
            rr       = reward / risk

            if rr < self.min_rr:
                return self._wait(f"BUY R:R {rr:.2f} below minimum {self.min_rr}.")

            pip_mult = self._pip_multiplier(close)
            sl_pips  = round(risk * pip_mult, 1)
            tp_pips  = round(reward * pip_mult, 1)

            confidence = self._score_bull(
                rsi, adx, plus_di, minus_di, vol_ratio, rr,
                confluence, hist_accel_up, bull_timing
            )

            reasons = [
                f"Momentum BUY — Supertrend bullish, ADX {adx:.1f}",
                f"+DI {plus_di:.1f} > −DI {minus_di:.1f}",
                f"RSI {rsi:.1f} resuming after pullback",
                f"MACD hist {'accelerating' if hist_accel_up else 'growing'}",
                f"Vol {vol_ratio:.1f}×",
                f"R:R {rr:.2f}:1 | SL {sl_pips}p → TP {tp_pips}p",
            ]
            if bull_timing:
                reasons.append("Stoch K crossed above D")
            if hidden_bull_div:
                reasons.append("[Hidden bull divergence — extra strength]")

            return {
                "action":     "BUY",
                "confidence": confidence,
                "reason":     " | ".join(reasons),
                "sl_pips":    sl_pips,
                "tp_pips":    tp_pips,
                "sl_price":   round(sl_price, 5),
                "tp_price":   round(tp_price, 5),
                "rr":         round(rr, 2),
                "confluence": confluence,
            }

        # ─────────────────────────────────────────────────────────────────────
        # SELL path
        # ─────────────────────────────────────────────────────────────────────
        bear_regime = (st_dir == -1) and (close < ema) and (minus_di > plus_di)
        bear_macd   = (macd_line < 0) and (macd_hist < prev_hist)
        bear_rsi    = rsi <= self.rsi_bear_max
        bear_timing = stoch_bear_cross and stoch_k > 20

        if bear_regime:
            if hidden_bull_div:
                return self._wait(
                    f"Hidden bullish divergence — price at lower low but RSI "
                    f"not confirming ({rsi:.1f}). Bearish momentum may be exhausted."
                )
            if not bear_macd:
                return self._wait(
                    f"Bear regime confirmed (ADX {adx:.1f}, Supertrend bearish, "
                    f"−DI > +DI) but MACD histogram not accelerating downward yet."
                )
            if not bear_rsi:
                return self._wait(
                    f"Bear regime present but RSI {rsi:.1f} > {self.rsi_bear_max}. "
                    f"Waiting for bearish momentum to establish."
                )
            if not had_bear_pullback:
                return self._wait(
                    f"Bear trend valid but no retracement in last {self.pullback_bars} bars "
                    f"(RSI max {float(rsi_window.max()):.1f} vs threshold {self.pullback_rsi_min}). "
                    f"Avoiding extended short entries — waiting for bounce."
                )

            swing_high = float(data["high"].iloc[-(self.pullback_bars + 2):-1].max())
            sl_price   = swing_high + (self.atr_sl_mult * atr)
            risk       = sl_price - close
            if risk <= 0:
                return self._wait("SELL SL calculation invalid (swing high below close).")

            confluence = self._count_bear_confluences(
                hist_accel_down, bear_timing, had_bear_pullback, hidden_bear_div, vol_ratio
            )
            tp_mult  = self.atr_tp_base_mult + (confluence / 5.0) * (self.atr_tp_max_mult - self.atr_tp_base_mult)
            tp_price = close - (tp_mult * atr)
            reward   = close - tp_price
            rr       = reward / risk

            if rr < self.min_rr:
                return self._wait(f"SELL R:R {rr:.2f} below minimum {self.min_rr}.")

            pip_mult = self._pip_multiplier(close)
            sl_pips  = round(risk * pip_mult, 1)
            tp_pips  = round(reward * pip_mult, 1)

            confidence = self._score_bear(
                rsi, adx, minus_di, plus_di, vol_ratio, rr,
                confluence, hist_accel_down, bear_timing
            )

            reasons = [
                f"Momentum SELL — Supertrend bearish, ADX {adx:.1f}",
                f"−DI {minus_di:.1f} > +DI {plus_di:.1f}",
                f"RSI {rsi:.1f} resuming after retracement",
                f"MACD hist {'accelerating down' if hist_accel_down else 'falling'}",
                f"Vol {vol_ratio:.1f}×",
                f"R:R {rr:.2f}:1 | SL {sl_pips}p → TP {tp_pips}p",
            ]
            if bear_timing:
                reasons.append("Stoch K crossed below D")
            if hidden_bear_div:
                reasons.append("[Hidden bear divergence confirmed]")

            return {
                "action":     "SELL",
                "confidence": confidence,
                "reason":     " | ".join(reasons),
                "sl_pips":    sl_pips,
                "tp_pips":    tp_pips,
                "sl_price":   round(sl_price, 5),
                "tp_price":   round(tp_price, 5),
                "rr":         round(rr, 2),
                "confluence": confluence,
            }

        return self._wait(
            f"No qualified momentum setup. "
            f"ADX {adx:.1f}, ST {'bull' if st_dir == 1 else 'bear'}, "
            f"RSI {rsi:.1f}, MACD {'above' if macd_line > 0 else 'below'} zero."
        )

    # ── Confluence counters ───────────────────────────────────────────────────

    def _count_bull_confluences(
        self, hist_accel: bool, stoch_cross: bool, had_pullback: bool,
        hidden_bull_div: bool, vol_ratio: float
    ) -> int:
        score = 0
        if hist_accel:         score += 1
        if stoch_cross:        score += 1
        if had_pullback:       score += 1
        if hidden_bull_div:    score += 1
        if vol_ratio >= 1.5:   score += 1
        return score

    def _count_bear_confluences(
        self, hist_accel: bool, stoch_cross: bool, had_pullback: bool,
        hidden_bear_div: bool, vol_ratio: float
    ) -> int:
        score = 0
        if hist_accel:         score += 1
        if stoch_cross:        score += 1
        if had_pullback:       score += 1
        if hidden_bear_div:    score += 1
        if vol_ratio >= 1.5:   score += 1
        return score

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _score_bull(
        self, rsi: float, adx: float, plus_di: float, minus_di: float,
        vol_ratio: float, rr: float, confluence: int,
        hist_accel: bool, stoch_cross: bool
    ) -> float:
        score = 0.50   # base — all gates have already passed

        # RSI depth above threshold (more momentum = higher score)
        score += min((rsi - self.rsi_bull_min) / 45.0, 1.0) * 0.10

        # ADX conviction
        score += min((adx - self.adx_min) / 35.0, 1.0) * 0.10

        # DI spread (directionality)
        di_spread = max(plus_di - minus_di, 0.0)
        score += min(di_spread / 25.0, 1.0) * 0.08

        # Volume surge
        if vol_ratio >= 2.0:   score += 0.08
        elif vol_ratio >= 1.5: score += 0.05
        else:                  score += 0.02

        # R:R quality (1.5→4.0 → 0→0.08)
        score += min((rr - self.min_rr) / 2.5, 1.0) * 0.08

        # Confluence count (0–5 → 0–0.08)
        score += (confluence / 5.0) * 0.08

        # Precision bonuses
        if hist_accel:   score += 0.04
        if stoch_cross:  score += 0.04

        return round(min(score, 0.97), 2)

    def _score_bear(
        self, rsi: float, adx: float, minus_di: float, plus_di: float,
        vol_ratio: float, rr: float, confluence: int,
        hist_accel: bool, stoch_cross: bool
    ) -> float:
        score = 0.50

        score += min((self.rsi_bear_max - rsi) / 45.0, 1.0) * 0.10
        score += min((adx - self.adx_min) / 35.0, 1.0) * 0.10

        di_spread = max(minus_di - plus_di, 0.0)
        score += min(di_spread / 25.0, 1.0) * 0.08

        if vol_ratio >= 2.0:   score += 0.08
        elif vol_ratio >= 1.5: score += 0.05
        else:                  score += 0.02

        score += min((rr - self.min_rr) / 2.5, 1.0) * 0.08
        score += (confluence / 5.0) * 0.08

        if hist_accel:   score += 0.04
        if stoch_cross:  score += 0.04

        return round(min(score, 0.97), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pip_multiplier(price: float) -> float:
        if price > 100:   # JPY pairs, Gold, indices
            return 100.0
        return 10000.0    # Standard 5-digit forex

    @staticmethod
    def _wait(reason: str) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": reason}