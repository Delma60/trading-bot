# strategies/arbitrage.py
import pandas as pd
import numpy as np

class ArbitrageStrategy:
    """
    Statistical Arbitrage (Pairs Trading).
    Dynamically finds highly correlated assets and bets on mean-reversion 
    when their historical spread diverges via Z-Score.
    """
    
    def __init__(self, z_score_threshold: float = 2.0, lookback: int = 20, min_correlation: float = 0.80):
        self.z_score_threshold = z_score_threshold
        self.lookback = lookback
        self.min_correlation = min_correlation
        
        # A broad universe of symbols to search for correlations.
        # In a real setup, this could be passed down from portfolio_config.json
        self.market_universe = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", 
            "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "US30", "NAS100"
        ]
        
        # Cache to store dynamically discovered pairs so we don't recalculate 
        # complex correlation matrices every 15 minutes.
        self.dynamic_pairs = {}

    def _find_correlated_pair(self, symbol: str, broker) -> str:
        """
        Scans the market universe to find the asset with the highest historical 
        price correlation to the target symbol.
        """
        # Return from cache if we already found the sister pair recently
        if symbol in self.dynamic_pairs:
            return self.dynamic_pairs[symbol]
            
        print(f"[Arbitrage Engine]: 🧠 Dynamically searching for a correlated pair for {symbol}...")
        
        # Fetch long-term baseline data (Daily timeframe) for the primary symbol
        primary_df = broker.get_historical_rates(symbol, timeframe="D1", count=100)
        if primary_df is None or primary_df.empty:
            return None

        best_pair = None
        highest_correlation = 0.0

        for candidate in self.market_universe:
            if candidate == symbol:
                continue
                
            candidate_df = broker.get_historical_rates(candidate, timeframe="D1", count=100)
            if candidate_df is None or candidate_df.empty:
                continue

            # Ensure dataframes are exactly the same length to prevent numpy errors
            min_len = min(len(primary_df), len(candidate_df))
            series_a = primary_df['close'].iloc[-min_len:]
            series_b = candidate_df['close'].iloc[-min_len:]
            
            # Calculate Pearson Correlation Coefficient
            corr = series_a.corr(series_b)
            
            if corr > highest_correlation and corr >= self.min_correlation:
                highest_correlation = corr
                best_pair = candidate

        if best_pair:
            print(f"[Arbitrage Engine]: ✅ Paired {symbol} with {best_pair} (Correlation: {highest_correlation*100:.1f}%)")
            self.dynamic_pairs[symbol] = best_pair
        else:
            print(f"[Arbitrage Engine]: ⚠️ No highly correlated pairs found for {symbol}.")
            # Cache a 'None' state so we don't spam the broker trying to find one every minute
            self.dynamic_pairs[symbol] = None 
            
        return best_pair

    def analyze(self, df: pd.DataFrame, symbol: str, broker) -> dict:
        """
        Calculates the Z-Score of the spread between dynamically correlated assets.
        """
        # 1. Dynamically find (or load) the correlated sister pair
        sister_symbol = self._find_correlated_pair(symbol, broker)
        if not sister_symbol:
            return {"action": "WAIT", "confidence": 0.0, "reason": "No valid statistical pair exists."}

        # 2. Fetch the sister pair's current operational timeframe data
        sister_df = broker.get_historical_rates(sister_symbol, timeframe="H1", count=self.lookback + 10)
        
        if sister_df is None or sister_df.empty or len(df) < self.lookback:
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Could not fetch operational data for {sister_symbol}."}

        # 3. Align the Data
        min_len = min(len(df), len(sister_df))
        asset_a_close = df['close'].iloc[-min_len:].values
        asset_b_close = sister_df['close'].iloc[-min_len:].values

        # Calculate the Spread Ratio
        spread = asset_a_close / asset_b_close

        # 4. Calculate the Z-Score
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(window=self.lookback).mean()
        rolling_std = spread_series.rolling(window=self.lookback).std()
        
        current_spread = spread_series.iloc[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]

        if pd.isna(current_std) or current_std == 0:
            return {"action": "WAIT", "confidence": 0.0, "reason": "Insufficient volatility to calculate Z-Score."}

        z_score = (current_spread - current_mean) / current_std

        # 5. The Logic Rules
        
        # BUY SIGNAL: Asset A is abnormally cheap compared to Asset B
        if z_score < -self.z_score_threshold:
            confidence = min(0.99, 0.75 + (abs(z_score) - self.z_score_threshold) * 0.1)
            return {
                "action": "BUY",
                "confidence": round(confidence, 2),
                "reason": f"Stat-Arb: {symbol} undervalued vs {sister_symbol} (Z: {z_score:.2f})"
            }

        # SELL SIGNAL: Asset A is abnormally expensive compared to Asset B
        elif z_score > self.z_score_threshold:
            confidence = min(0.99, 0.75 + (z_score - self.z_score_threshold) * 0.1)
            return {
                "action": "SELL",
                "confidence": round(confidence, 2),
                "reason": f"Stat-Arb: {symbol} overvalued vs {sister_symbol} (Z: {z_score:.2f})"
            }

        return {"action": "WAIT", "confidence": 0.0, "reason": f"Spread is normal (Z: {z_score:.2f})"}