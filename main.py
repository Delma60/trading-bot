from trader import Trader
from chat import Chatbot
from strategies.strategy_manager import StrategyManager

login = 106683365
password = "@oIt7uPp"


if __name__ == "__main__":
    chat = Chatbot(
        intents_filepath="intents.json", 
        broker= Trader(),
        strategy_manager= StrategyManager()
    )
    
    chat.start_chat()
