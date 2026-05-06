# strategies/mean_reversion.py
import pandas as pd
import pandas_ta as ta

class MeanReversionStrategy:
    """
    Looks for prices that have stretched too far from their average 
    and bets on them snapping back to the middle.
    """
    
    def __init__(self, bb_length: int = 20, bb_std: float = 2.0, rsi_length: int = 14):
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_length = rsi_length

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Takes a dataframe of price history and returns a trading signal.
        """
        if df is None or len(df) < self.bb_length:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough data"}

        # 1. Calculate Bollinger Bands
        # Returns a DataFrame with 'BBL_20_2.0' (Lower), 'BBM_20_2.0' (Mid), 'BBU_20_2.0' (Upper)
        bbands = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        if bbands is None:
             return {"action": "WAIT", "confidence": 0.0, "reason": "Indicator calculation failed"}
             
        df = pd.concat([df, bbands], axis=1)

        # 2. Calculate RSI
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_length)

        # 3. Get the most recent completed candle (iloc[-1] is the live ticking candle, iloc[-2] is the last closed one)
        # Using the last closed candle is safer to avoid repainting signals.
        latest = df.iloc[-2]

        # Dynamically find the column names pandas-ta generated
        lower_band_col = f"BBL_{self.bb_length}_{self.bb_std}"
        upper_band_col = f"BBU_{self.bb_length}_{self.bb_std}"
        
        close_price = latest['close']
        rsi = latest['RSI']
        lower_band = latest[lower_band_col]
        upper_band = latest[upper_band_col]

        # 4. The Logic Rules
        
        # BUY SIGNAL: Price pierced the bottom band AND RSI is deeply oversold
        if close_price < lower_band and rsi < 30.0:
            # Confidence increases the lower the RSI gets below 30
            confidence = min(0.99, 0.70 + ((30.0 - rsi) / 100.0))
            return {
                "action": "BUY", 
                "confidence": round(confidence, 2), 
                "reason": f"Oversold Bounce (RSI: {rsi:.1f}, Price below Lower BB)"
            }

        # SELL SIGNAL: Price pierced the top band AND RSI is deeply overbought
        elif close_price > upper_band and rsi > 70.0:
            # Confidence increases the higher the RSI gets above 70
            confidence = min(0.99, 0.70 + ((rsi - 70.0) / 100.0))
            return {
                "action": "SELL", 
                "confidence": round(confidence, 2), 
                "reason": f"Overbought Rejection (RSI: {rsi:.1f}, Price above Upper BB)"
            }

        # DEFAULT: Nothing interesting is happening
        return {"action": "WAIT", "confidence": 0.0, "reason": "Price is ranging inside the bands."}