from trader import Trader
import pandas as pd
import numpy as np
import pandas_ta as pta

backcandle = 15
win = backcandle + 1

td = Trader()
td.connect(106683365, "@oIt7uPp", "MetaQuotes-Demo")
ticker = td.get_symbol_info("EURUSD")
ticker_data = td.get_tick_data("EURUSD", num_ticks=5)
df = td.ohclv_data("EURUSD", timeframe="H1", num_bars=1000)

# breakout test strategy

df['rsi'] = pta.rsi(df['close'], length=12)
df['ema'] = pta.ema(df['close'], length=150)
df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=14)

above = (np.minimum(df['open'], df['close']) > df['ema']).astype(int)
below = (np.minimum(df['open'], df['close']) < df['ema']).astype(int)

upt = (above.rolling(win, min_periods=win).sum() == win)
dnt = (below.rolling(win, min_periods=win).sum() == win)

signal = np.zeros(len(df), dtype=int)
signal[upt & dnt] = 3
signal[upt & ~dnt] = 2
signal[dnt & ~upt] = 1

df['ema_signal'] = signal

# mark pivot function

def mark_pivots(df:pd.DataFrame, window:int=5, high_col="high", low_col="low"):
    span = 2 * window + 1
    roll_max = df[high_col].rolling(span, center=True).max()
    roll_min = df[low_col].rolling(span, center=True).min()
    
    pivot_high = (df[high_col] >= roll_max) & (roll_max.notna())
    pivot_low = (df[low_col] <= roll_min) & (roll_min.notna())
    
    pivots = np.zeros(len(df), dtype=int)
    pivots[pivot_high] += 1
    pivots[pivot_low] += 2
    return pd.Series(pivots, index=df.index, name="isPivot")

df['isPivot'] = mark_pivots(df)


def detect_structure(candle:int, backcandle:int=40, window:int=5):
    if candle - backcandle < 0:
        return 0, None
    
    prev_bar = candle - 1
    if prev_bar  < 0:
        return 0, None
    price_df = df.iloc[candle-backcandle:candle]
    close_now = df.loc[candle, "close"]
    close_prev = df.loc[prev_bar, "close"]
    
    piv_df = df.iloc[candle - backcandle : candle - window + 1]
    
    ph_df = piv_df[piv_df['isPivot'] == 1]
    
    if not ph_df.empty:
        ph_idx = ph_df.index[-1]
        ph_val = ph_df.loc[ph_idx, 'high']
        if close_now > ph_val and close_prev <= ph_val:
            pl_before = piv_df.loc[:ph_idx - 1]
            pl_before = pl_before[pl_before['isPivot'] == 2]
            
            pl_after = piv_df.loc[ph_idx + 1:]
            pl_after = pl_after[pl_after['isPivot'] == 2]
            
            if not pl_before.empty and not pl_after.empty:
                pl1_val = pl_before.iloc[-1]['low']
                pl2_val = pl_after['low'].min()
                
                if pl2_val < pl1_val:
                    return 2, ph_idx
    
    pl_df = piv_df[piv_df['isPivot'] == 2]
    if not pl_df.empty:
        pl_idx = pl_df.index[-1]
        pl_val = pl_df.loc[pl_idx, 'low']
        if close_now < pl_val and close_prev >= pl_val:
            ph_before = piv_df.loc[:pl_idx - 1]
            ph_before = ph_before[ph_before['isPivot'] == 1]
            
            ph_after = piv_df.loc[pl_idx + 1:]
            ph_after = ph_after[ph_after['isPivot'] == 1]
            
            if not ph_before.empty and not ph_after.empty:
                ph1_val = ph_before.iloc[-1]['high']
                ph2_val = ph_after['high'].max()
                
                if ph2_val > ph1_val:
                    return 1, pl_idx
    
    return 0, None
    

start_index = 0 
end_index = len(df)
for candle in range(start_index, end_index):
    signal, ref_idx = detect_structure(candle)
    df.at[candle, 'breakout_signal'] = signal


df['breakout_signal'] = df.apply(lambda row: row['breakout_signal'] if row['ema_signal'] == row['breakout_signal'] else 0, axis=1)
