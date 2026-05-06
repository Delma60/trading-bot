# strategies/momentum.py
import pandas as pd
import pandas_ta as ta

class MomentumStrategy:
    """
    Looks for assets that are moving strongly in one direction 
    and bets on the continuation of that movement.
    """
    
    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, ema_length: int = 20, rsi_length: int = 14):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_length = ema_length
        self.rsi_length = rsi_length

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Takes a dataframe of price history and returns a trading signal.
        """
        if df is None or len(df) < self.macd_slow + self.macd_signal:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough data"}

        # 1. Calculate MACD
        # Returns a DataFrame with MACD line, Histogram, and Signal line
        macd_df = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd_df is None:
             return {"action": "WAIT", "confidence": 0.0, "reason": "MACD calculation failed"}
             
        df = pd.concat([df, macd_df], axis=1)

        # 2. Calculate EMA and RSI
        df['EMA'] = ta.ema(df['close'], length=self.ema_length)
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_length)

        # 3. Get the most recent closed candle
        latest = df.iloc[-2]
        previous = df.iloc[-3] # We look one candle back to check if momentum is *increasing*

        # Dynamically find the MACD column names
        macd_line_col = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        macd_hist_col = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        macd_sig_col = f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        
        close = latest['close']
        ema = latest['EMA']
        rsi = latest['RSI']
        
        macd_line = latest[macd_line_col]
        macd_sig = latest[macd_sig_col]
        macd_hist = latest[macd_hist_col]
        prev_macd_hist = previous[macd_hist_col]

        # 4. The Logic Rules
        
        # BUY SIGNAL: 
        # Price is above EMA (Uptrend) AND MACD crossed above Signal AND Histogram is growing AND RSI shows strength
        if close > ema and macd_line > macd_sig and macd_line > 0 and macd_hist > prev_macd_hist and rsi > 55.0:
            
            # Confidence grows the steeper the RSI is and the wider the MACD gap
            confidence = min(0.99, 0.70 + ((rsi - 50.0) / 100.0))
            return {
                "action": "BUY", 
                "confidence": round(confidence, 2), 
                "reason": f"Bullish Momentum (RSI: {rsi:.1f}, MACD Expanding)"
            }

        # SELL SIGNAL: 
        # Price is below EMA (Downtrend) AND MACD crossed below Signal AND Histogram is dropping AND RSI shows weakness
        elif close < ema and macd_line < macd_sig and macd_line < 0 and macd_hist < prev_macd_hist and rsi < 45.0:
            
            # Confidence grows the lower the RSI gets below 50
            confidence = min(0.99, 0.70 + ((50.0 - rsi) / 100.0))
            return {
                "action": "SELL", 
                "confidence": round(confidence, 2), 
                "reason": f"Bearish Momentum (RSI: {rsi:.1f}, MACD Expanding)"
            }

        # DEFAULT: Nothing interesting is happening
        return {"action": "WAIT", "confidence": 0.0, "reason": "No clear momentum established."}
    
    