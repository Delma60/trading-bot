# strategies/trend_following.py
import pandas as pd
import pandas_ta as ta


class TrendFollowingStrategy:
    """
    Rides long-term macro trends using SMA crossover + ADX + MACD + OBV volume.

    Improvements over v1
    --------------------
    * DataFrame mutation fix — v1 wrote columns directly into the caller's df;
                               this version works on an internal copy throughout.
    * MACD confirmation     — MACD histogram must agree with the trade direction,
                               preventing entries on stale/decaying momentum.
    * DI spread conviction  — the gap between +DI and −DI is used to scale
                               confidence, not just raw ADX.
    * Cross-age filter      — detects how many bars ago the SMA cross occurred.
                               Very fresh crosses (< min_cross_age) are skipped
                               to avoid whipsaws; very old ones get a confidence
                               discount as the trade is already mature.
    * OBV trend filter      — on-balance volume confirms that money-flow agrees
                               with the price trend.
    * Graduated confidence  — pulls from ADX strength, DI spread, MACD power,
                               and OBV agreement instead of a single formula.
    """

    def __init__(
        self,
        sma_fast:      int   = 50,
        sma_slow:      int   = 200,
        adx_length:    int   = 14,
        adx_threshold: float = 25.0,
        min_cross_age: int   = 3,     # bars; skip if cross is too fresh (whipsaw)
        max_cross_age: int   = 50,    # bars; discount if cross is too old
    ):
        self.sma_fast      = sma_fast
        self.sma_slow      = sma_slow
        self.adx_length    = adx_length
        self.adx_threshold = adx_threshold
        self.min_cross_age = min_cross_age
        self.max_cross_age = max_cross_age

    def analyze(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < self.sma_slow + self.adx_length + 10:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": f"Not enough data for {self.sma_slow}-bar SMA."}

        data = df.copy()   # ← never mutate the caller's DataFrame

        # ── Indicators ──────────────────────────────────────────────────────
        data["sma_fast"] = ta.sma(data["close"], length=self.sma_fast)
        data["sma_slow"] = ta.sma(data["close"], length=self.sma_slow)

        # ADX + directional indicators
        adx_df = ta.adx(data["high"], data["low"], data["close"],
                        length=self.adx_length)
        if adx_df is None or adx_df.empty:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": "ADX calculation failed."}
        data = pd.concat([data, adx_df], axis=1)
        adx_col      = f"ADX_{self.adx_length}"
        plus_di_col  = f"DMP_{self.adx_length}"
        minus_di_col = f"DMN_{self.adx_length}"

        # MACD (12/26/9 standard)
        macd_df = ta.macd(data["close"])
        if macd_df is not None and not macd_df.empty:
            data = pd.concat([data, macd_df], axis=1)
            macd_hist_col = [c for c in macd_df.columns if "h" in c.lower()][0]
        else:
            data["macd_hist"] = 0.0
            macd_hist_col = "macd_hist"

        # OBV (on-balance volume trend)
        data["obv"]        = ta.obv(data["close"], data["volume"])
        data["obv_sma"]    = ta.sma(data["obv"], length=20)
        data["obv_rising"] = (data["obv"] > data["obv_sma"]).astype(int)

        data.dropna(inplace=True)
        if data.empty:
            return {"action": "WAIT", "confidence": 0.0,
                    "reason": "Insufficient data after indicator warm-up."}

        # ── Snapshot ─────────────────────────────────────────────────────────
        bar      = data.iloc[-2]   # last confirmed closed candle
        close    = bar["close"]
        sma_fast = bar["sma_fast"]
        sma_slow = bar["sma_slow"]
        adx      = bar[adx_col]
        plus_di  = bar[plus_di_col]
        minus_di = bar[minus_di_col]
        macd_h   = bar[macd_hist_col]
        obv_up   = bool(bar["obv_rising"])

        # ── Cross-age detection ──────────────────────────────────────────────
        # Count how many consecutive bars the fast SMA has been above (or below) slow SMA.
        above_series = data["sma_fast"] > data["sma_slow"]
        cross_age    = self._consecutive_bars(above_series, above_series.iloc[-2])

        # ── Gate: ADX must confirm a real trend ─────────────────────────────
        if adx < self.adx_threshold:
            return {
                "action":     "WAIT",
                "confidence": 0.0,
                "reason":     (f"Market is ranging/choppy "
                               f"(ADX: {adx:.1f} < {self.adx_threshold}). "
                               f"Waiting for trend to develop."),
            }

        # ── BUY ─────────────────────────────────────────────────────────────
        bull_cross  = sma_fast > sma_slow
        bull_price  = close > sma_fast           # price hasn't run away from MA
        bull_macd   = macd_h > 0                 # histogram positive = rising momentum
        bull_di     = plus_di > minus_di

        if bull_cross and bull_price and bull_di:
            # Whipsaw guard: cross too fresh
            if cross_age < self.min_cross_age:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Golden cross only {cross_age} bar(s) old — "
                                  f"waiting for confirmation (min: {self.min_cross_age})."}

            if not bull_macd:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Bull structure present but MACD histogram is negative "
                                  f"({macd_h:.5f}) — momentum not confirmed."}

            confidence = self._score(adx, plus_di, minus_di, cross_age, obv_up)
            return {
                "action":     "BUY",
                "confidence": confidence,
                "reason": (
                    f"Bull Trend Confirmed — ADX: {adx:.1f}, "
                    f"+DI: {plus_di:.1f} vs −DI: {minus_di:.1f}, "
                    f"MACD hist: {macd_h:+.5f}, "
                    f"Cross age: {cross_age} bars, "
                    f"OBV: {'rising ✓' if obv_up else 'weak'}"
                ),
            }

        # ── SELL ─────────────────────────────────────────────────────────────
        bear_cross = sma_fast < sma_slow
        bear_price = close < sma_fast
        bear_macd  = macd_h < 0
        bear_di    = minus_di > plus_di

        if bear_cross and bear_price and bear_di:
            if cross_age < self.min_cross_age:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Death cross only {cross_age} bar(s) old — "
                                  f"waiting for confirmation (min: {self.min_cross_age})."}

            if not bear_macd:
                return {"action": "WAIT", "confidence": 0.0,
                        "reason": f"Bear structure present but MACD histogram is positive "
                                  f"({macd_h:.5f}) — downward momentum not confirmed."}

            confidence = self._score(adx, minus_di, plus_di, cross_age, not obv_up)
            return {
                "action":     "SELL",
                "confidence": confidence,
                "reason": (
                    f"Bear Trend Confirmed — ADX: {adx:.1f}, "
                    f"−DI: {minus_di:.1f} vs +DI: {plus_di:.1f}, "
                    f"MACD hist: {macd_h:+.5f}, "
                    f"Cross age: {cross_age} bars, "
                    f"OBV: {'falling ✓' if not obv_up else 'weak'}"
                ),
            }

        return {
            "action":     "WAIT",
            "confidence": 0.0,
            "reason":     "Trend structure exists but price/MACD conditions not met.",
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _score(
        self,
        adx:       float,
        di_lead:   float,   # dominant DI (+DI for bull, −DI for bear)
        di_lag:    float,   # weaker DI
        cross_age: int,
        obv_agrees: bool,
    ) -> float:
        """
        Compose a [0.50, 0.97] confidence score from multiple inputs.

        Contributions
        -------------
        ADX strength (30–60 range mapped to 0–0.15)
        DI spread    (higher spread = more one-sided trend)
        OBV confirms (+0.06 bonus)
        Cross maturity (discount if > max_cross_age)
        """
        # ADX component: 0 at threshold, maxes out around ADX=60
        adx_score = min((adx - self.adx_threshold) / 35.0, 1.0) * 0.15

        # DI spread component
        di_spread     = max(di_lead - di_lag, 0.0)
        di_score      = min(di_spread / 30.0, 1.0) * 0.12

        # OBV bonus
        obv_bonus = 0.06 if obv_agrees else 0.0

        # Maturity discount for very old crosses (trade already partially played out)
        maturity_discount = 0.0
        if cross_age > self.max_cross_age:
            maturity_discount = min((cross_age - self.max_cross_age) / 50.0, 0.10)

        raw = 0.60 + adx_score + di_score + obv_bonus - maturity_discount
        return round(min(max(raw, 0.50), 0.97), 2)

    @staticmethod
    def _consecutive_bars(series: pd.Series, current_state: bool) -> int:
        """
        Count how many consecutive bars at the tail of ``series`` share
        the same boolean value as ``current_state``.
        """
        count = 0
        for val in reversed(series.values):
            if bool(val) == current_state:
                count += 1
            else:
                break
        return count