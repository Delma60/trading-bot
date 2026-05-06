import pandas as pd
from strategies.arbitrage import ArbitrageStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from strategies.scalping import ScalpingStrategy
from strategies.trend_following import TrendFollowingStrategy
from trader import Trader
from .mean_reversion import MeanReversionStrategy

class DummyStrategy:
    """A safe placeholder for strategies you haven't fully coded yet."""
    def analyze(self, df: pd.DataFrame) -> dict:
        return {"action": "WAIT", "confidence": 0.0, "reason": "Strategy logic not yet implemented."}

class StrategyManager:
    # Use exact casing that the Portfolio Manager expects
    strategies = [
        "Mean_Reversion",
        "Momentum",
        "Breakout",
        "Scalping",
        "News_Trading",
        "Sentiment_Analysis",
        "Arbitrage",
    ]
    
    # Keep descriptions separate from the actual strategy engines
    descriptions = {
        "Mean_Reversion": "Mean Reversion: Buys oversold dips, sells overbought rips.",
        "Momentum": "Momentum: Buys when price is trending upward, sells when trending downward.",
        "Breakout": "Breakout: Buys when price breaks above resistance, sells when breaks below support.",
        "Scalping": "Scalping: Makes small profits from frequent, short-term trades.",
        "News_Trading": "News Trading: Trades based on economic news and events.",
        "Sentiment_Analysis": "Sentiment Analysis: Trades based on market sentiment.",
        "Arbitrage": "Arbitrage: Exploits price differences between markets."
    }
    
    def __init__(self, broker: Trader):
        self.broker = broker
        
        # Instantiate engines ONCE to save memory, rather than recreating them every scan
        self.engines = {
            "Mean_Reversion": MeanReversionStrategy(),
            "Momentum": MomentumStrategy(),
            "Breakout": BreakoutStrategy(),
            "Scalping": ScalpingStrategy(),
            "News_Trading": DummyStrategy(),
            "Sentiment_Analysis": DummyStrategy(),
            "Arbitrage": ArbitrageStrategy(),
            "Trend_Following": TrendFollowingStrategy()
        }
    
    def get_strategy_description(self, strategy_name: str) -> str:
        return self.descriptions.get(strategy_name, "Unknown strategy description.")
    
    def execute_strategy(self, strategy_name: str):
        """Fetches the pre-loaded strategy engine object."""
        description = self.get_strategy_description(strategy_name)
        # print(f"[Strategy Manager]: Executing {strategy_name} -> {description}")
        
        # Return the actual engine object (or a safe Dummy if it fails)
        return self.engines.get(strategy_name, DummyStrategy())
    
    def check_signals(self, symbol: str, strategy: str = "Mean_Reversion", timeframe: str = "H1") -> dict:
        """Fetches the latest data and asks the chosen strategy what to do."""
        
        # 1. Ensure the requested strategy actually exists
        if strategy not in self.engines.keys():
            print(f"[Strategy Manager]: ⚠️ Unknown strategy '{strategy}'. Defaulting to Mean_Reversion.")
            strategy = "Mean_Reversion"
            
        # 2. Get the correct engine object
        engine = self.execute_strategy(strategy)

        # 3. Fetch the raw data required to make a decision
        df = self.broker.get_historical_rates(symbol, timeframe=timeframe, count=100)
        
        if df is None or df.empty:
            return {"action": "WAIT", "confidence": 0.0, "reason": f"Could not fetch data for {symbol}"}

        # 4. Hand the data to the Strategy Engine and get the result
        signal = engine.analyze(df)
        
        return signal