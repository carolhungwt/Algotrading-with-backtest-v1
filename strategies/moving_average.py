import pandas as pd
import numpy as np
from strategy_manager import Strategy

class SimpleMovingAverageCrossover(Strategy):
    """
    A strategy based on the crossover of two simple moving averages.
    
    Generates buy signals when the shorter moving average crosses above the longer
    moving average, and sell signals when the shorter moving average crosses below
    the longer moving average.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters (dict): Strategy parameters
                - short_window (int): Period of the shorter moving average (default: 50)
                - long_window (int): Period of the longer moving average (default: 200)
        """
        default_params = {
            'short_window': 50,
            'long_window': 200
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        super().__init__(self.parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (pandas.DataFrame): Historical price data with at least a 'Close' column
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Get parameters
        short_window = self.parameters['short_window']
        long_window = self.parameters['long_window']
        
        # Create a copy of the data
        df = data.copy()
        
        # Generate moving averages
        df['SMA_short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        
        # Initialize signal series
        signals = pd.Series(0, index=df.index)
        
        # Generate signals based on crossovers
        signals[(df['SMA_short'] > df['SMA_long']) & 
                (df['SMA_short'].shift(1) <= df['SMA_long'].shift(1))] = 1  # Buy signal
        
        signals[(df['SMA_short'] < df['SMA_long']) & 
                (df['SMA_short'].shift(1) >= df['SMA_long'].shift(1))] = -1  # Sell signal
        
        return signals
    
    def __str__(self):
        return f"Simple Moving Average Crossover (Short: {self.parameters['short_window']}, Long: {self.parameters['long_window']})" 