import pandas as pd
import numpy as np
from strategy_manager import Strategy

class TripleMASlope(Strategy):
    """
    A strategy based on three moving averages with slope conditions.
    
    Buy signal: 10MA crosses above 20MA (10MA slope must be positive)
    Sell signal: 6MA crosses below 10MA (6MA slope must be negative)
    No signal if the slopes of all 3 MAs are similar (market is equivocal/ranging)
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters (dict): Strategy parameters
                - fast_window (int): Period of the fast moving average (default: 6)
                - mid_window (int): Period of the mid moving average (default: 10)
                - slow_window (int): Period of the slow moving average (default: 20)
                - slope_period (int): Period to calculate slope over (default: 5)
                - slope_threshold (float): Threshold for determining if slopes are similar (default: 0.1)
        """
        default_params = {
            'fast_window': 6,
            'mid_window': 10,
            'slow_window': 20,
            'slope_period': 5,
            'slope_threshold': 0.1  # If slope differences are less than this, consider them "same"
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        super().__init__(self.parameters)
    
    def calculate_slope(self, series, period):
        """
        Calculate the slope of a series over the specified period.
        
        Args:
            series (pandas.Series): Data series
            period (int): Period to calculate slope over
            
        Returns:
            pandas.Series: Slope values
        """
        # Simple linear regression slope
        # We use the last 'period' values to calculate the slope
        slope = pd.Series(0, index=series.index)
        
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            
            # Linear regression: y = mx + b
            m, b = np.polyfit(x, y, 1)
            slope.iloc[i] = m
            
        return slope
    
    def generate_signals(self, data):
        """
        Generate trading signals based on triple MA crossover with slope conditions.
        
        Args:
            data (pandas.DataFrame): Historical price data with at least a 'Close' column
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Get parameters
        fast_window = self.parameters['fast_window']
        mid_window = self.parameters['mid_window']
        slow_window = self.parameters['slow_window']
        slope_period = self.parameters['slope_period']
        slope_threshold = self.parameters['slope_threshold']
        
        # Create a copy of the data
        df = data.copy()
        
        # Generate moving averages
        df['MA_fast'] = df['Close'].rolling(window=fast_window, min_periods=1).mean()
        df['MA_mid'] = df['Close'].rolling(window=mid_window, min_periods=1).mean()
        df['MA_slow'] = df['Close'].rolling(window=slow_window, min_periods=1).mean()
        
        # Calculate slopes
        df['Slope_fast'] = self.calculate_slope(df['MA_fast'], slope_period)
        df['Slope_mid'] = self.calculate_slope(df['MA_mid'], slope_period)
        df['Slope_slow'] = self.calculate_slope(df['MA_slow'], slope_period)
        
        # Initialize signal series
        signals = pd.Series(0, index=df.index)
        
        # Check for equivocal market (all slopes are similar)
        slopes_similar = (
            (abs(df['Slope_fast'] - df['Slope_mid']) < slope_threshold) &
            (abs(df['Slope_mid'] - df['Slope_slow']) < slope_threshold) &
            (abs(df['Slope_fast'] - df['Slope_slow']) < slope_threshold)
        )
        
        # Generate buy signals: 10MA crosses above 20MA with positive 10MA slope
        buy_condition = (
            (~slopes_similar) &  # Not in equivocal market
            (df['MA_mid'] > df['MA_slow']) &  # 10MA above 20MA
            (df['MA_mid'].shift(1) <= df['MA_slow'].shift(1)) &  # Crossover just happened
            (df['Slope_mid'] > 0)  # 10MA has positive slope
        )
        
        # Generate sell signals: 6MA crosses below 10MA with negative 6MA slope
        sell_condition = (
            (~slopes_similar) &  # Not in equivocal market
            (df['MA_fast'] < df['MA_mid']) &  # 6MA below 10MA
            (df['MA_fast'].shift(1) >= df['MA_mid'].shift(1)) &  # Crossover just happened
            (df['Slope_fast'] < 0)  # 6MA has negative slope
        )
        
        # Apply signals
        signals[buy_condition] = 1  # Buy signal
        signals[sell_condition] = -1  # Sell signal
        
        return signals
    
    def __str__(self):
        return (f"Triple MA Slope Strategy (Fast: {self.parameters['fast_window']}, "
                f"Mid: {self.parameters['mid_window']}, Slow: {self.parameters['slow_window']})") 