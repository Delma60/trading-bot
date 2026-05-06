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

        try:
            # 1. Calculate Bollinger Bands
            bbands = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
            if bbands is None or len(bbands) == 0:
                return {"action": "WAIT", "confidence": 0.0, "reason": "Indicator calculation failed"}
            
            df = pd.concat([df, bbands], axis=1)

            # 2. Calculate RSI
            df['RSI'] = ta.rsi(df['close'], length=self.rsi_length)

            # 3. Get the most recent completed candle
            latest = df.iloc[-2]

            # 4. Dynamically find the correct column names pandas_ta generated
            # pandas_ta bbands returns columns like: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            bbl_cols = [col for col in df.columns if col.startswith('BBL_')]
            bbu_cols = [col for col in df.columns if col.startswith('BBU_')]
            
            if not bbl_cols or not bbu_cols:
                return {"action": "WAIT", "confidence": 0.0, "reason": "Bollinger Bands columns not found"}
            
            lower_band_col = bbl_cols[-1]  # Get the most recent BBL column
            upper_band_col = bbu_cols[-1]  # Get the most recent BBU column
            
            close_price = latest['close']
            rsi = latest['RSI']
            lower_band = latest[lower_band_col]
            upper_band = latest[upper_band_col]

            # 5. The Logic Rules
            # BUY SIGNAL: Price pierced the bottom band AND RSI is deeply oversold
            if close_price < lower_band and rsi < 30.0:
                confidence = min(0.99, 0.70 + ((30.0 - rsi) / 100.0))
                return {
                    "action": "BUY", 
                    "confidence": round(confidence, 2), 
                    "reason": f"Oversold Bounce (RSI: {rsi:.1f}, Price below Lower BB)"
                }

            # SELL SIGNAL: Price pierced the top band AND RSI is deeply overbought
            elif close_price > upper_band and rsi > 70.0:
                confidence = min(0.99, 0.70 + ((rsi - 70.0) / 100.0))
                return {
                    "action": "SELL", 
                    "confidence": round(confidence, 2), 
                    "reason": f"Overbought Rejection (RSI: {rsi:.1f}, Price above Upper BB)"
                }

            # DEFAULT: Nothing interesting is happening
            print(f"[Mean Reversion Strategy]: Price is within bands. Close: {close_price:.4f}, Lower BB: {lower_band:.4f}, Upper BB: {upper_band:.4f}, RSI: {rsi:.1f}")
            return {"action": "WAIT", "confidence": 0.0, "reason": "Price is ranging inside the bands."}
            
        except KeyError as e:
            print(f"[Mean Reversion Strategy]: KeyError - Missing expected column: {e}")
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Missing indicator data: {e}"}
        except Exception as e:
            print(f"[Mean Reversion Strategy]: Exception - Unexpected error: {e}")
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Analysis error: {e}"}