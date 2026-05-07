import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Transforms raw OHLCV data into a rich feature matrix for ML/DL models.
    Every feature is derived purely from OHLCV — no external feeds required.

    Call:
        feat_df = FeatureEngineer.compute(raw_df)

    The returned DataFrame has all original columns plus the engineered ones,
    with leading NaN rows (from rolling windows) already dropped.
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()

        # ── Price & Return Features ──────────────────────────────────────────
        f["return_1"]   = f["close"].pct_change(1)
        f["return_3"]   = f["close"].pct_change(3)
        f["return_5"]   = f["close"].pct_change(5)
        f["return_10"]  = f["close"].pct_change(10)
        f["log_return"] = np.log(f["close"] / f["close"].shift(1))

        # ── Volatility ───────────────────────────────────────────────────────
        f["volatility_10"] = f["log_return"].rolling(10).std()
        f["volatility_20"] = f["log_return"].rolling(20).std()
        f["atr"]           = FeatureEngineer._atr(f, 14)
        f["atr_ratio"]     = f["atr"] / f["close"]          # normalised ATR

        # ── Moving Averages & Distance ───────────────────────────────────────
        for p in [5, 10, 20, 50, 200]:
            f[f"sma_{p}"] = f["close"].rolling(p).mean()

        f["ema_9"]  = f["close"].ewm(span=9,  adjust=False).mean()
        f["ema_21"] = f["close"].ewm(span=21, adjust=False).mean()
        f["ema_55"] = f["close"].ewm(span=55, adjust=False).mean()

        # Normalised distance from price to key MAs
        f["dist_sma20"]  = (f["close"] - f["sma_20"])  / f["sma_20"]
        f["dist_sma50"]  = (f["close"] - f["sma_50"])  / f["sma_50"]
        f["dist_sma200"] = (f["close"] - f["sma_200"]) / f["sma_200"]

        # Cross signals (binary)
        f["golden_cross"] = (f["sma_50"]  > f["sma_200"]).astype(int)
        f["ema_cross"]    = (f["ema_9"]   > f["ema_21"]).astype(int)

        # ── Momentum Oscillators ─────────────────────────────────────────────
        f["rsi_14"] = FeatureEngineer._rsi(f["close"], 14)
        f["rsi_7"]  = FeatureEngineer._rsi(f["close"],  7)

        macd_line, macd_signal, macd_hist = FeatureEngineer._macd(f["close"])
        f["macd"]        = macd_line
        f["macd_signal"] = macd_signal
        f["macd_hist"]   = macd_hist
        f["macd_cross"]  = (macd_line > macd_signal).astype(int)

        f["stoch_k"], f["stoch_d"] = FeatureEngineer._stochastic(f, 14, 3)
        f["williams_r"]            = FeatureEngineer._williams_r(f, 14)
        f["cci"]                   = FeatureEngineer._cci(f, 20)

        # ── Trend Strength (ADX) ─────────────────────────────────────────────
        f["adx"], f["plus_di"], f["minus_di"] = FeatureEngineer._adx(f, 14)
        f["di_diff"] = f["plus_di"] - f["minus_di"]    # directional bias

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_mid, bb_upper, bb_lower = FeatureEngineer._bollinger(f["close"], 20, 2)
        f["bb_mid"]   = bb_mid
        f["bb_upper"] = bb_upper
        f["bb_lower"] = bb_lower
        f["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)          # squeeze
        f["bb_pos"]   = (f["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 0=bottom 1=top

        # Ensure we have a volume series available for feature engineering
        if 'volume' not in f.columns:
            if 'tick_volume' in f.columns:
                f['volume'] = f['tick_volume']
            elif 'real_volume' in f.columns:
                f['volume'] = f['real_volume']
            else:
                raise KeyError("OHLCV data must include 'volume', 'tick_volume', or 'real_volume'.")

        # ── Volume Features ──────────────────────────────────────────────────
        f["volume_sma_20"] = f["volume"].rolling(20).mean()
        f["volume_ratio"]  = f["volume"] / (f["volume_sma_20"] + 1e-9)   # >1 = above-avg
        f["obv"]           = FeatureEngineer._obv(f)
        f["obv_sma_10"]    = f["obv"].rolling(10).mean()
        f["obv_trend"]     = (f["obv"] > f["obv_sma_10"]).astype(int)

        # Rolling VWAP (20-bar approximation)
        f["vwap"]      = (f["close"] * f["volume"]).rolling(20).sum() / (f["volume"].rolling(20).sum() + 1e-9)
        f["dist_vwap"] = (f["close"] - f["vwap"]) / (f["vwap"] + 1e-9)

        # ── Candlestick Body / Shadow ────────────────────────────────────────
        f["body"]         = f["close"] - f["open"]
        f["body_abs"]     = f["body"].abs()
        f["upper_shadow"] = f["high"] - f[["close", "open"]].max(axis=1)
        f["lower_shadow"] = f[["close", "open"]].min(axis=1) - f["low"]
        f["body_ratio"]   = f["body_abs"] / (f["high"] - f["low"] + 1e-9)
        f["is_bullish"]   = (f["close"] > f["open"]).astype(int)

        # ── Market Regime ────────────────────────────────────────────────────
        f["regime_trending"] = (f["adx"] > 25).astype(int)
        f["regime_bullish"]  = (f["close"] > f["sma_50"]).astype(int)

        f.dropna(inplace=True)
        return f

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast    = series.ewm(span=fast,   adjust=False).mean()
        ema_slow    = series.ewm(span=slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram   = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        low_min  = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_max = df["high"].rolling(period).max()
        low_min  = df["low"].rolling(period).min()
        return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-9)

    @staticmethod
    def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-9)

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14):
        up       = df["high"].diff()
        down     = -df["low"].diff()
        plus_dm  = up.where((up > down)   & (up   > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        atr      = FeatureEngineer._atr(df, period)
        plus_di  = 100 * plus_dm.rolling(period).mean()  / (atr + 1e-9)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-9)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx      = dx.rolling(period).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def _bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
        mid   = series.rolling(period).mean()
        std   = series.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return mid, upper, lower

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["close"].diff()).fillna(0)
        return (direction * df["volume"]).cumsum()