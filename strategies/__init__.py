# This file is required to make Python treat the directory as a package.
from strategy_manager import Strategy

# Import all strategies for easier access
from strategies.moving_average import SimpleMovingAverageCrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.composite import CompositeStrategy
from strategies.triple_ma_crossover import TripleMASlope

__all__ = [
    'Strategy',
    'SimpleMovingAverageCrossover',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerBandsStrategy',
    'CompositeStrategy',
    'TripleMASlope'
] 