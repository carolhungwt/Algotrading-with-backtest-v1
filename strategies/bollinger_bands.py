import pandas as pd
import numpy as np
from strategy_manager import Strategy

class BollingerBandsStrategy(Strategy):
    """
    A strategy based on Bollinger Bands.
    
    Generates buy signals when the price touches or crosses below the lower band
    and sell signals when the price touches or crosses above the upper band.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters (dict): Strategy parameters
                - window (int): Window for the moving average (default: 20)
                - num_std (float): Number of standard deviations for the bands (default: 2.0)
        """
        default_params = {
            'window': 20,
            'num_std': 2.0
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        super().__init__(self.parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data (pandas.DataFrame): Historical price data with at least a 'Close' column
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Get parameters
        window = self.parameters['window']
        num_std = self.parameters['num_std']
        
        # Create a copy of the data
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['MA'] = df['Close'].rolling(window=window, min_periods=1).mean()
        df['Std'] = df['Close'].rolling(window=window, min_periods=1).std()
        
        df['Upper'] = df['MA'] + (df['Std'] * num_std)
        df['Lower'] = df['MA'] - (df['Std'] * num_std)
        
        # Initialize signal series
        signals = pd.Series(0, index=df.index)
        
        # Generate signals based on price crossing the bands
        # Buy when price crosses below lower band
        signals[(df['Close'] <= df['Lower']) & 
                (df['Close'].shift(1) > df['Lower'].shift(1))] = 1
        
        # Sell when price crosses above upper band
        signals[(df['Close'] >= df['Upper']) & 
                (df['Close'].shift(1) < df['Upper'].shift(1))] = -1
        
        # Alternative: Buy when price crosses back above lower band (coming back into the range)
        # signals[(df['Close'] > df['Lower']) & (df['Close'].shift(1) <= df['Lower'].shift(1))] = 1
        
        # Alternative: Sell when price crosses back below upper band (coming back into the range)
        # signals[(df['Close'] < df['Upper']) & (df['Close'].shift(1) >= df['Upper'].shift(1))] = -1
        
        return signals
    
    def __str__(self):
        return f"Bollinger Bands Strategy (Window: {self.parameters['window']}, StdDev: {self.parameters['num_std']})" 