import pandas as pd
import numpy as np
import pandas_ta as pta

class BreakoutStrategy:
    """
    Advanced Structure Breakout Strategy with Liquidity Sweep Detection.
    Looks for a sweep of pivot highs/lows followed by a structural break,
    filtered by a strong EMA trend.
    """
    def __init__(self, window: int = 5, backcandle: int = 40, ema_length: int = 150):
        self.window = window
        self.backcandle = backcandle
        self.ema_length = ema_length
        self.trend_win = 16  # window + 1 from your script

    def mark_pivots(self, df: pd.DataFrame, high_col="high", low_col="low") -> pd.Series:
        """
        Marks Pivot Highs (1) and Pivot Lows (2).
        Uses center=True but is safely isolated from lookahead bias in the analyze() method.
        """
        span = 2 * self.window + 1
        roll_max = df[high_col].rolling(span, center=True).max()
        roll_min = df[low_col].rolling(span, center=True).min()
        
        pivot_high = (df[high_col] >= roll_max) & (roll_max.notna())
        pivot_low = (df[low_col] <= roll_min) & (roll_min.notna())
        
        pivots = np.zeros(len(df), dtype=int)
        pivots[pivot_high] = 1
        pivots[pivot_low] = 2
        return pd.Series(pivots, index=df.index)

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        The main execution method called by StrategyManager.
        Returns a dictionary with the action, confidence, and trade parameters.
        """
        # 1. Ensure we have enough data to calculate EMA and lookbacks
        if len(df) < self.ema_length + self.trend_win + self.backcandle:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Insufficient data."}

        # Work on a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # 2. Calculate Indicators
        df['ema'] = pta.ema(df['close'], length=self.ema_length)
        df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=14)
        df['isPivot'] = self.mark_pivots(df)

        # 3. Calculate Trend Filter (Only needs to evaluate the recent past)
        # np.minimum finds the bottom of the candle body
        min_body = np.minimum(df['open'], df['close'])
        above_ema = (min_body > df['ema']).astype(int)
        below_ema = (min_body < df['ema']).astype(int)

        uptrend = (above_ema.rolling(self.trend_win, min_periods=self.trend_win).sum() == self.trend_win)
        downtrend = (below_ema.rolling(self.trend_win, min_periods=self.trend_win).sum() == self.trend_win)

        # 4. Extract current live states
        current_idx = len(df) - 1
        prev_idx = current_idx - 1
        
        close_now = df['close'].iloc[current_idx]
        close_prev = df['close'].iloc[prev_idx]
        current_atr = df['atr'].iloc[current_idx]
        
        is_uptrend = uptrend.iloc[current_idx]
        is_downtrend = downtrend.iloc[current_idx]

        # 5. Safe Pivot Slicing (Prevents Lookahead Bias)
        # We only look at pivots that have fully closed and passed the `window` threshold
        safe_end_idx = current_idx - self.window + 1
        safe_start_idx = current_idx - self.backcandle
        
        if safe_start_idx < 0:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Not enough pivot history."}
            
        piv_df = df.iloc[safe_start_idx : safe_end_idx]

        # ==========================================
        # STRATEGY LOGIC: LONG BREAKOUT (Sweep & Break)
        # ==========================================
        if is_uptrend:
            ph_df = piv_df[piv_df['isPivot'] == 1]
            if not ph_df.empty:
                ph_idx = ph_df.index[-1]
                ph_val = ph_df.loc[ph_idx, 'high']
                
                # Did we cross the pivot high on THIS specific candle?
                if close_now > ph_val and close_prev <= ph_val:
                    # Liquidity Sweep Check
                    pl_before = piv_df.loc[:ph_idx - 1]
                    pl_before = pl_before[pl_before['isPivot'] == 2]
                    
                    pl_after = piv_df.loc[ph_idx + 1:]
                    pl_after = pl_after[pl_after['isPivot'] == 2]
                    
                    if not pl_before.empty and not pl_after.empty:
                        pl1_val = pl_before.iloc[-1]['low']
                        pl2_val = pl_after['low'].min()
                        
                        # If recent low swept the older low, valid structure break!
                        if pl2_val < pl1_val:
                            # Calculate dynamic risk
                            stop_loss_price = pl2_val - (current_atr * 0.5) # Place SL slightly below the sweep
                            take_profit_price = close_now + ((close_now - stop_loss_price) * 2) # 1:2 R:R
                            
                            return {
                                "action": "BUY", 
                                "confidence": 0.85, 
                                "reason": f"Structure Breakout above {ph_val:.5f} after Liquidity Sweep.",
                                "suggested_sl": stop_loss_price,
                                "suggested_tp": take_profit_price
                            }

        # ==========================================
        # STRATEGY LOGIC: SHORT BREAKOUT (Sweep & Break)
        # ==========================================
        elif is_downtrend:
            pl_df = piv_df[piv_df['isPivot'] == 2]
            if not pl_df.empty:
                pl_idx = pl_df.index[-1]
                pl_val = pl_df.loc[pl_idx, 'low']
                
                # Did we cross the pivot low on THIS specific candle?
                if close_now < pl_val and close_prev >= pl_val:
                    # Liquidity Sweep Check
                    ph_before = piv_df.loc[:pl_idx - 1]
                    ph_before = ph_before[ph_before['isPivot'] == 1]
                    
                    ph_after = piv_df.loc[pl_idx + 1:]
                    ph_after = ph_after[ph_after['isPivot'] == 1]
                    
                    if not ph_before.empty and not ph_after.empty:
                        ph1_val = ph_before.iloc[-1]['high']
                        ph2_val = ph_after['high'].max()
                        
                        # If recent high swept the older high, valid structure break!
                        if ph2_val > ph1_val:
                            # Calculate dynamic risk
                            stop_loss_price = ph2_val + (current_atr * 0.5) # Place SL slightly above the sweep
                            take_profit_price = close_now - ((stop_loss_price - close_now) * 2) # 1:2 R:R
                            
                            return {
                                "action": "SELL", 
                                "confidence": 0.85, 
                                "reason": f"Structure Breakdown below {pl_val:.5f} after Liquidity Sweep.",
                                "suggested_sl": stop_loss_price,
                                "suggested_tp": take_profit_price
                            }

        return {"action": "WAIT", "confidence": 0.0, "reason": "No breakout structure detected."}