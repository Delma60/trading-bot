# strategies/trend_following.py
import pandas as pd
import pandas_ta as ta

class TrendFollowingStrategy:
    """
    Rides long-term macro trends. Uses the 50/200 SMA crossover to determine direction,
    and requires an ADX > 25 to confirm the trend actually has strength.
    """
    
    def __init__(self, sma_fast: int = 50, sma_slow: int = 200, adx_length: int = 14, adx_threshold: float = 25.0):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Takes a dataframe of price history and returns a trading signal.
        Trend following works best on higher timeframes (H1, H4, D1).
        """
        if df is None or len(df) < self.sma_slow + 5:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough data for 200 SMA."}

        # 1. Calculate Simple Moving Averages (The Trend Direction)
        df['SMA_Fast'] = ta.sma(df['close'], length=self.sma_fast)
        df['SMA_Slow'] = ta.sma(df['close'], length=self.sma_slow)

        # 2. Calculate ADX (The Trend Strength)
        # pandas_ta returns ADX, DPM (Positive Direction), and DMN (Negative Direction)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=self.adx_length)
        if adx_df is None:
             return {"action": "WAIT", "confidence": 0.0, "reason": "ADX calculation failed."}
             
        df = pd.concat([df, adx_df], axis=1)

        # 3. Get the latest closed candle
        latest = df.iloc[-2]

        # Dynamically find the ADX column name
        adx_col = f"ADX_{self.adx_length}"
        
        close = latest['close']
        sma_fast = latest['SMA_Fast']
        sma_slow = latest['SMA_Slow']
        adx = latest[adx_col]

        # 4. The Logic Rules
        
        # BUY SIGNAL: Fast SMA > Slow SMA (Golden Cross) AND ADX > Threshold (Strong Trend) AND Price > Fast SMA (Pullback protection)
        if sma_fast > sma_slow and adx > self.adx_threshold and close > sma_fast:
            
            # Confidence scales with ADX strength. An ADX of 50 is a monstrously strong trend.
            confidence = min(0.99, 0.70 + ((adx - self.adx_threshold) / 100.0))
            return {
                "action": "BUY", 
                "confidence": round(confidence, 2), 
                "reason": f"Bull Trend Confirmed (ADX: {adx:.1f}, Price > 50 SMA)"
            }

        # SELL SIGNAL: Fast SMA < Slow SMA (Death Cross) AND ADX > Threshold AND Price < Fast SMA
        elif sma_fast < sma_slow and adx > self.adx_threshold and close < sma_fast:
            
            # Confidence scales with ADX strength.
            confidence = min(0.99, 0.70 + ((adx - self.adx_threshold) / 100.0))
            return {
                "action": "SELL", 
                "confidence": round(confidence, 2), 
                "reason": f"Bear Trend Confirmed (ADX: {adx:.1f}, Price < 50 SMA)"
            }

        # DEFAULT: ADX is too low, or price is chopped between the moving averages.
        if adx < self.adx_threshold:
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Market is ranging/choppy (ADX: {adx:.1f} < {self.adx_threshold})."}
        
        return {"action": "WAIT", "confidence": 0.0, "reason": "Trend exists, but price action is pulling back."}