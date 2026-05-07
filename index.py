from trader import Trader
import pandas as pd
import numpy as np
import pandas_ta as pta

td = Trader()
td.connect(106683365, "@oIt7uPp", "MetaQuotes-Demo")
ticker = td.get_symbol_info("EURUSD")
ticker_data = td.get_tick_data("EURUSD", num_ticks=5)
ohclv = td.ohclv_data("EURUSD", timeframe="H1", num_bars=1000)
df = pd.DataFrame([ticker_data])

# Keep OHLCV columns for indicator calculation
# Drop only any extra non-price columns if present
# ohclv = ohclv.drop(columns=['tick_volume', 'spread', 'real_volume'], errors='ignore')
# sma
ohclv['sma'] = pta.sma(ohclv['close'], length=21)

ohclv['simple_returns'] = ohclv['close'].pct_change()
ohclv['cumulative_returns'] = (1 + ohclv['simple_returns']).cumprod() - 1
ohclv['log_returns'] = np.log(1 + ohclv['simple_returns'])
ohclv['ratios'] = ohclv['close'] / ohclv['sma']
ohclv.dropna(inplace=True)
# percentiles
percentiles = [15, 20,  50, 80, 85]
ratios = ohclv['ratios']
percentile_values = np.percentile(ratios, percentiles)
print(percentile_values)
sell = percentiles[-1]
buy = percentiles[0]
# positions

ohclv['position'] = np.where(ohclv['ratios'] > sell, -1, np.where(ohclv['ratios'] < buy, 1, 0))

print(ohclv[['close', 'sma', 'ratios', 'simple_returns', 'cumulative_returns', 'log_returns', 'position']].tail())
# print(ohclv[['close', 'sma', 'ratios']].tail(20))
# print(ohclv['ratios'].describe())
