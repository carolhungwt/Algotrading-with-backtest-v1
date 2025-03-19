import json
import importlib
import inspect
import logging
import os
import sys
from typing import Dict, List, Any, Type

# Abstract base strategy class
class Strategy:
    """Base strategy class that all trading strategies should inherit from."""
    
    def __init__(self, parameters=None):
        """
        Initialize strategy with optional parameters.
        
        Args:
            parameters (dict): Strategy parameters
        """
        self.parameters = parameters or {}
        self.name = self.__class__.__name__
        
    def generate_signals(self, data):
        """
        Generate trading signals from data.
        
        Args:
            data (pandas.DataFrame): Historical price data
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def set_parameters(self, parameters):
        """
        Update strategy parameters.
        
        Args:
            parameters (dict): New parameters
        """
        self.parameters.update(parameters)
    
    def __str__(self):
        return f"{self.name} with parameters: {self.parameters}"


class StrategyManager:
    """Manages the loading and creation of trading strategies."""
    
    def __init__(self):
        """Initialize the StrategyManager."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.available_strategies = self._discover_strategies()
        
    def _discover_strategies(self):
        """
        Discover all strategy classes in the 'strategies' directory.
        
        Returns:
            dict: Dictionary mapping strategy names to their classes
        """
        strategies = {}
        
        # First check if the strategies directory exists, if not create it
        if not os.path.exists("strategies"):
            os.makedirs("strategies")
            
            # Create an __init__.py file in the strategies directory
            with open("strategies/__init__.py", "w") as f:
                f.write("# This file is required to make Python treat the directory as a package.")
        
        # Ensure strategies directory is in the path
        strategies_dir = os.path.abspath("strategies")
        if strategies_dir not in sys.path:
            sys.path.insert(0, strategies_dir)
        
        # Look for strategies in the strategies directory
        for filename in os.listdir(strategies_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"strategies.{module_name}")
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, Strategy) 
                            and obj != Strategy and name != "Strategy"):
                            strategies[name] = obj
                except Exception as e:
                    self.logger.error(f"Error loading strategy module {module_name}: {str(e)}")
        
        # Add built-in strategies
        from strategies.moving_average import SimpleMovingAverageCrossover
        from strategies.rsi import RSIStrategy
        from strategies.macd import MACDStrategy
        from strategies.bollinger_bands import BollingerBandsStrategy
        
        strategies["SimpleMovingAverageCrossover"] = SimpleMovingAverageCrossover
        strategies["RSIStrategy"] = RSIStrategy
        strategies["MACDStrategy"] = MACDStrategy
        strategies["BollingerBandsStrategy"] = BollingerBandsStrategy
        
        self.logger.info(f"Discovered {len(strategies)} strategies: {', '.join(strategies.keys())}")
        return strategies
    
    def get_available_strategies(self):
        """
        Get a list of all available strategy names.
        
        Returns:
            list: List of strategy names
        """
        return list(self.available_strategies.keys())
    
    def load_strategies(self, strategy_names, params_json=None):
        """
        Load specified strategies with optional parameters.
        
        Args:
            strategy_names (list): List of strategy names to load
            params_json (str): JSON string containing parameters for strategies
            
        Returns:
            list: List of strategy instances
        """
        strategies = []
        
        # Parse parameters if provided
        params = {}
        if params_json and params_json.strip():
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing strategy parameters: {str(e)}")
        
        # Create strategy instances
        for name in strategy_names:
            if name in self.available_strategies:
                try:
                    # Get strategy class
                    strategy_class = self.available_strategies[name]
                    
                    # Get parameters for this strategy if available
                    strategy_params = params.get(name, {})
                    
                    # Create strategy instance
                    strategy = strategy_class(parameters=strategy_params)
                    strategies.append(strategy)
                    
                    self.logger.info(f"Loaded strategy: {strategy}")
                except Exception as e:
                    self.logger.error(f"Error creating strategy {name}: {str(e)}")
            else:
                self.logger.warning(f"Strategy {name} not found")
        
        return strategies
    
    def combine_strategies(self, strategies, weighting=None):
        """
        Create a composite strategy that combines multiple strategies.
        
        Args:
            strategies (list): List of strategy instances
            weighting (list): Optional list of weights for each strategy
            
        Returns:
            CompositeStrategy: A strategy that combines signals from multiple strategies
        """
        from strategies.composite import CompositeStrategy
        return CompositeStrategy(strategies, weighting) 