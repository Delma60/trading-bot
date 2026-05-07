# strategies/mean_reversion.py
import pandas as pd
import pandas_ta as ta
import numpy as np

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
        df['sma'] = ta.sma(df['close'], length=21)

        df['simple_returns'] = df['close'].pct_change()
        df['cumulative_returns'] = (1 + df['simple_returns']).cumprod() - 1
        df['log_returns'] = np.log(1 + df['simple_returns'])
        df['ratios'] = df['close'] / df['sma']
        df.dropna(inplace=True)
        # percentiles
        percentiles = [15, 20,  50, 80, 85]
        ratios = df['ratios']
        percentile_values = np.percentile(ratios, percentiles)
        print(percentile_values)
        sell = percentiles[-1]
        buy = percentiles[0]
        # positions

        df['position'] = np.where(df['ratios'] > sell, -1, np.where(df['ratios'] < buy, 1, 0))

        print(df[['close', 'sma', 'ratios', 'simple_returns', 'cumulative_returns', 'log_returns', 'position']].tail())
        return {}