import pandas as pd
import numpy as np
from strategy_manager import Strategy

class RSIStrategy(Strategy):
    """
    A strategy based on the Relative Strength Index (RSI) indicator.
    
    Generates buy signals when RSI falls below the oversold threshold and
    sell signals when RSI rises above the overbought threshold.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters (dict): Strategy parameters
                - period (int): Period for RSI calculation (default: 14)
                - oversold (int): Oversold threshold (default: 30)
                - overbought (int): Overbought threshold (default: 70)
        """
        default_params = {
            'period': 14,
            'oversold': 30,
            'overbought': 70
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        super().__init__(self.parameters)
    
    def _calculate_rsi(self, prices, period):
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices (pandas.Series): Price series
            period (int): RSI period
            
        Returns:
            pandas.Series: RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI values.
        
        Args:
            data (pandas.DataFrame): Historical price data with at least a 'Close' column
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Get parameters
        period = self.parameters['period']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        
        # Create a copy of the data
        df = data.copy()
        
        # Calculate RSI
        df['RSI'] = self._calculate_rsi(df['Close'], period)
        
        # Initialize signal series
        signals = pd.Series(0, index=df.index)
        
        # Generate signals based on RSI thresholds
        # Buy when RSI crosses below oversold threshold
        signals[(df['RSI'] < oversold) & 
                (df['RSI'].shift(1) >= oversold)] = 1
        
        # Sell when RSI crosses above overbought threshold
        signals[(df['RSI'] > overbought) & 
                (df['RSI'].shift(1) <= overbought)] = -1
        
        return signals
    
    def __str__(self):
        return f"RSI Strategy (Period: {self.parameters['period']}, Oversold: {self.parameters['oversold']}, Overbought: {self.parameters['overbought']})" 