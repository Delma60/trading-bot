from manager.risk_manager import RiskManager
from trader import Trader
from chat import Chatbot
from strategies.strategy_manager import StrategyManager
from manager.portfolio_manager import PortfolioManager

login = 106683365
password = "@oIt7uPp"


if __name__ == "__main__":
    broker = Trader()

    # 2. Initialize the Engines
    strategy_manager = StrategyManager(broker)
    risk_manager = RiskManager(broker)
    portfolio_manager = PortfolioManager(broker, strategy_manager, risk_manager)
    
    bot = Chatbot(
        intents_filepath="data/intents.json", 
        broker=broker, 
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager
    )
    
    bot.start_chat()
