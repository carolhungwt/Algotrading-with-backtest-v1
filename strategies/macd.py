import pandas as pd
import numpy as np
from strategy_manager import Strategy

class MACDStrategy(Strategy):
    """
    A strategy based on the Moving Average Convergence Divergence (MACD) indicator.
    
    Generates buy signals when the MACD line crosses above the signal line and
    sell signals when the MACD line crosses below the signal line.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters (dict): Strategy parameters
                - fast_period (int): Period for the fast EMA (default: 12)
                - slow_period (int): Period for the slow EMA (default: 26)
                - signal_period (int): Period for the signal line EMA (default: 9)
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        super().__init__(self.parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD crossovers.
        
        Args:
            data (pandas.DataFrame): Historical price data with at least a 'Close' column
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Get parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        
        # Create a copy of the data
        df = data.copy()
        
        # Calculate MACD components
        df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line = Fast EMA - Slow EMA
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        
        # Signal line = EMA of MACD line
        df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # MACD histogram = MACD line - Signal line
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # Initialize signal series
        signals = pd.Series(0, index=df.index)
        
        # Generate signals based on MACD crossovers
        # Buy when MACD crosses above the signal line
        signals[(df['MACD'] > df['Signal']) & 
                (df['MACD'].shift(1) <= df['Signal'].shift(1))] = 1
        
        # Sell when MACD crosses below the signal line
        signals[(df['MACD'] < df['Signal']) & 
                (df['MACD'].shift(1) >= df['Signal'].shift(1))] = -1
        
        return signals
    
    def __str__(self):
        return f"MACD Strategy (Fast: {self.parameters['fast_period']}, Slow: {self.parameters['slow_period']}, Signal: {self.parameters['signal_period']})" 