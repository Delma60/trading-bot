# strategies/breakout.py
import pandas as pd
import pandas_ta as ta

class BreakoutStrategy:
    """
    Looks for price to break through historical support or resistance levels
    with high volatility to confirm a true breakout.
    """
    
    def __init__(self, lookback_window: int = 20, atr_length: int = 14):
        self.lookback_window = lookback_window
        self.atr_length = atr_length

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Takes a dataframe of price history and returns a trading signal.
        """
        if df is None or len(df) < self.lookback_window + 1:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough data"}

        # 1. Calculate the "Box" (Support and Resistance)
        # We use .shift(1) because we want the high/low of the PREVIOUS 20 candles. 
        # If we include the current candle, it will always equal the current price during a breakout!
        df['Resistance'] = df['high'].rolling(window=self.lookback_window).max().shift(1)
        df['Support'] = df['low'].rolling(window=self.lookback_window).min().shift(1)

        # 2. Calculate Volatility (ATR)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)

        # 3. Get the most recent closed candle
        latest = df.iloc[-2]
        
        close = latest['close']
        resistance = latest['Resistance']
        support = latest['Support']
        atr = latest['ATR']
        
        # Calculate the size of the breakout candle's body
        candle_body_size = abs(latest['close'] - latest['open'])

        # 4. The Logic Rules
        
        # BUY SIGNAL: Price closed above Resistance AND it's a strong candle
        if close > resistance and candle_body_size > (atr * 0.5):
            
            # Confidence grows the further it breaks past resistance
            distance_broken = close - resistance
            confidence = min(0.99, 0.75 + (distance_broken / atr))
            
            return {
                "action": "BUY", 
                "confidence": round(confidence, 2), 
                "reason": f"Bullish Breakout (Broke {self.lookback_window}-period high)"
            }

        # SELL SIGNAL: Price closed below Support AND it's a strong candle
        elif close < support and candle_body_size > (atr * 0.5):
            
            # Confidence grows the further it breaks past support
            distance_broken = support - close
            confidence = min(0.99, 0.75 + (distance_broken / atr))
            
            return {
                "action": "SELL", 
                "confidence": round(confidence, 2), 
                "reason": f"Bearish Breakout (Broke {self.lookback_window}-period low)"
            }

        # DEFAULT: Price is still stuck inside the box
        return {"action": "WAIT", "confidence": 0.0, "reason": "Price consolidating inside the range."}
    