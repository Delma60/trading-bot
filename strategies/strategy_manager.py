

class StrategyManager:
    strategies = [
        "mean_reversion",
        "momentum",
        "breakout",
        "scalping",
        "news_trading",
        "sentiment_analysis",
        "arbitrage",
    ]
    
    def compile_mean_reversion(self):
        return "Mean Reversion: Buys when price is below average, sells when above."

    def compile_momentum(self):
        return "Momentum: Buys when price is trending upward, sells when trending downward."

    def compile_breakout(self):
        return "Breakout: Buys when price breaks above resistance, sells when breaks below support."

    def compile_scalping(self):
        return "Scalping: Makes small profits from frequent, short-term trades."

    def compile_news_trading(self):
        return "News Trading: Trades based on economic news and events."

    def compile_sentiment_analysis(self):
        return "Sentiment Analysis: Trades based on market sentiment and情绪."

    def compile_arbitrage(self):
        return "Arbitrage: Exploits price differences between markets."
    
    def get_strategy_description(self, strategy_name):
        method_name = f"compile_{strategy_name}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method()
        else:
            return "Unknown strategy."
    
    def execute_strategy(self, strategy_name, broker):
        description = self.get_strategy_description(strategy_name)
        print(f"Executing {strategy_name} strategy: {description}")
        return self["compile_" + strategy_name]()