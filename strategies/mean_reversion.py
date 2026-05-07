import pandas as pd
import pandas_ta as ta
import numpy as np

class MeanReversionStrategy:
    """
    Looks for prices that have stretched too far from their average 
    and bets on them snapping back to the middle using Bollinger Bands and RSI.
    """
    
    def __init__(self, 
                 bb_length: int = 20, 
                 bb_std: float = 2.0, 
                 rsi_length: int = 14,
                 sma_length: int = 21,
                 lookback_window: int = 100,
                 percentile_lower: float = 0.15,
                 percentile_upper: float = 0.85):
        
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_length = rsi_length
        self.sma_length = sma_length
        self.lookback_window = lookback_window
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        
    def analyze(self, df: pd.DataFrame) -> dict:
        # 1. Safety Checks & Immutability
        if df is None or len(df) < max(self.lookback_window, self.bb_length, self.rsi_length, self.sma_length):
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough data points."}
        
        # Work on a copy to prevent SettingWithCopyWarnings and mutating the bot's global state
        data = df.copy()

        try:
            # 2. Calculate Indicators
            # Bollinger Bands
            bb = ta.bbands(data['close'], length=self.bb_length, std=self.bb_std)
            if bb is None or bb.empty:
                return {"action": "WAIT", "confidence": 0.0, "reason": "Failed to calculate Bollinger Bands."}
            
            data = pd.concat([data, bb], axis=1)
            
            # Dynamically identify BB column names to prevent floating-point string formatting errors
            bb_cols = bb.columns
            bb_lower = [col for col in bb_cols if col.startswith('BBL')][0]
            bb_upper = [col for col in bb_cols if col.startswith('BBU')][0]
                
            # RSI & SMA
            data['rsi'] = ta.rsi(data['close'], length=self.rsi_length)
            data['sma'] = ta.sma(data['close'], length=self.sma_length)
            data['ratios'] = data['close'] / data['sma']
            
            # --- FIX: Rolling Percentiles (Prevents Lookahead Bias) ---
            # Instead of np.percentile on the whole dataset, we use a rolling window
            data['buy_threshold'] = data['ratios'].rolling(window=self.lookback_window).quantile(self.percentile_lower)
            data['sell_threshold'] = data['ratios'].rolling(window=self.lookback_window).quantile(self.percentile_upper)
            
            # Clean NaNs caused by indicator windows and rolling lookbacks
            data.dropna(inplace=True)
            if data.empty:
                return {"action": "WAIT", "confidence": 0.0, "reason": "Data dropped to empty after lookbacks."}

            # --- SIGNAL GENERATION LOGIC ---
            last_row = data.iloc[-1]
            close = last_row['close']
            rsi = last_row['rsi']
            ratio = last_row['ratios']
            lower_band = last_row[bb_lower]
            upper_band = last_row[bb_upper]
            buy_threshold = last_row['buy_threshold']
            sell_threshold = last_row['sell_threshold']

            action = "WAIT"
            confidence = 0.0
            reason = "Price is within normal bounds."

            # CONFLUENCE 1: Bollinger Band Breakout + RSI Oversold/Overbought
            if close < lower_band and rsi < 30:
                action = "BUY"
                confidence = 0.85
                reason = f"Price ({close:.4f}) dropped below Lower BB ({lower_band:.4f}) with Oversold RSI ({rsi:.1f})."
                
            elif close > upper_band and rsi > 70:
                action = "SELL"
                confidence = 0.85
                reason = f"Price ({close:.4f}) broke above Upper BB ({upper_band:.4f}) with Overbought RSI ({rsi:.1f})."

            # CONFLUENCE 2: Backup logic using Rolling Ratio
            elif ratio < buy_threshold and rsi < 40:
                action = "BUY"
                confidence = 0.60
                reason = f"Price stretched below SMA (Ratio: {ratio:.3f} < {buy_threshold:.3f}) with RSI ({rsi:.1f})."
                
            elif ratio > sell_threshold and rsi > 60:
                action = "SELL"
                confidence = 0.60
                reason = f"Price stretched above SMA (Ratio: {ratio:.3f} > {sell_threshold:.3f}) with RSI ({rsi:.1f})."

            return {"action": action, "confidence": confidence, "reason": reason}

        except Exception as e:
            # Prevent crashes in a live environment
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Calculation Error: {str(e)}"}