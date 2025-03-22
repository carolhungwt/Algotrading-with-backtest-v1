import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
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
                - debug (bool): Enable debug mode for visualizations and detailed logging (default: False)
                - debug_dir (str): Directory to save debug files (default: 'debug')
        """
        default_params = {
            'fast_window': 6,
            'mid_window': 10,
            'slow_window': 20,
            'slope_period': 2,
            'slope_threshold': 0.2,  # If slope differences are less than this, consider them "same"
            'debug': False,
            'debug_dir': 'debug'
        }
        
        # Use provided parameters or defaults
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        
        # Create debug directory if debug mode is enabled
        if self.parameters['debug']:
            debug_dir = self.parameters['debug_dir']
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            if not os.path.exists(f"{debug_dir}/images"):
                os.makedirs(f"{debug_dir}/images")
            if not os.path.exists(f"{debug_dir}/csv"):
                os.makedirs(f"{debug_dir}/csv")
        
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
        # Create a Series with float dtype explicitly
        slope = pd.Series(0.0, index=series.index, dtype=float)
        
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            
            # Linear regression: y = mx + b
            m, b = np.polyfit(x, y, 1)
            slope.iloc[i] = float(m)  # Explicitly cast to float
            
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
        debug_mode = self.parameters['debug']
        
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
        
        # Add equivocal market indicator to dataframe
        df['Equivocal'] = slopes_similar
        
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
        
        # Add signals to dataframe for debugging
        df['Signal'] = signals
        
        # If debug mode is enabled, generate debug information
        if debug_mode:
            self._generate_debug_output(df, data)
        
        return signals
    
    def _generate_debug_output(self, df, original_data):
        """
        Generate debug output including visualizations and CSV files.
        
        Args:
            df (pandas.DataFrame): DataFrame with calculated indicators and signals
            original_data (pandas.DataFrame): Original price data
        """
        try:
            debug_dir = self.parameters['debug_dir']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed data to CSV
            csv_path = f"{debug_dir}/csv/triple_ma_debug_{timestamp}.csv"
            df.to_csv(csv_path)
            print(f"Saved debug data to {csv_path}")
            
            # Generate special CSV just for trades
            self._save_trades_data(df, timestamp)
            
            # Generate debug visualizations
            self._generate_debug_charts(df, original_data, timestamp)
        except Exception as e:
            print(f"ERROR in debug output generation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _save_trades_data(self, df, timestamp):
        """
        Save trade-specific data to CSV.
        
        Args:
            df (pandas.DataFrame): DataFrame with calculated indicators and signals
            timestamp (str): Timestamp for filename
        """
        try:
            debug_dir = self.parameters['debug_dir']
            
            # Find all buy and sell signals
            buys = df[df['Signal'] == 1].copy()
            sells = df[df['Signal'] == -1].copy()
            
            # Create trades dataframe with important information
            if not buys.empty or not sells.empty:
                trades = pd.concat([buys, sells])
                trades = trades.sort_index()
                
                # Add context of signal
                trades['Signal_Type'] = trades['Signal'].map({1: 'BUY', -1: 'SELL'})
                
                # Calculate slope differences for analysis
                trades['Slope_Fast_Mid_Diff'] = trades['Slope_fast'] - trades['Slope_mid']
                trades['Slope_Mid_Slow_Diff'] = trades['Slope_mid'] - trades['Slope_slow']
                
                # Extract relevant columns for trade analysis
                trade_cols = [
                    'Signal_Type', 'Close', 
                    'MA_fast', 'MA_mid', 'MA_slow',
                    'Slope_fast', 'Slope_mid', 'Slope_slow',
                    'Equivocal', 'Slope_Fast_Mid_Diff', 'Slope_Mid_Slow_Diff'
                ]
                
                trade_data = trades[trade_cols]
                trade_csv_path = f"{debug_dir}/csv/triple_ma_trades_{timestamp}.csv"
                trade_data.to_csv(trade_csv_path)
                print(f"Saved trade details to {trade_csv_path}")
            else:
                print("No trades to save - no buy or sell signals found")
        except Exception as e:
            print(f"ERROR in _save_trades_data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_debug_charts(self, df, original_data, timestamp):
        """
        Generate debug charts showing MAs, slopes, and signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with calculated indicators and signals
            original_data (pandas.DataFrame): Original price data
            timestamp (str): Timestamp for filename
        """
        try:
            debug_dir = self.parameters['debug_dir']
            
            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot price and MAs on top subplot
            ax1.plot(df.index, original_data['Close'], label='Price', alpha=0.5, color='black')
            ax1.plot(df.index, df['MA_fast'], label=f"{self.parameters['fast_window']}MA", color='blue')
            ax1.plot(df.index, df['MA_mid'], label=f"{self.parameters['mid_window']}MA", color='green')
            ax1.plot(df.index, df['MA_slow'], label=f"{self.parameters['slow_window']}MA", color='red')
            
            # Highlight buy signals
            buys = df[df['Signal'] == 1]
            if not buys.empty:
                ax1.scatter(buys.index, buys['Close'], marker='^', color='green', s=100, label='Buy Signal')
            
            # Highlight sell signals
            sells = df[df['Signal'] == -1]
            if not sells.empty:
                ax1.scatter(sells.index, sells['Close'], marker='v', color='red', s=100, label='Sell Signal')
            
            # Highlight equivocal regions
            equivocal_regions = df[df['Equivocal'] == True]
            if not equivocal_regions.empty:
                ax1.axvspan(equivocal_regions.index[0], equivocal_regions.index[-1], 
                          alpha=0.2, color='gray', label='Equivocal Market')
            
            ax1.set_title('Price and Moving Averages with Trade Signals')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot slopes on bottom subplot
            ax2.plot(df.index, df['Slope_fast'], label=f"{self.parameters['fast_window']}MA Slope", color='blue')
            ax2.plot(df.index, df['Slope_mid'], label=f"{self.parameters['mid_window']}MA Slope", color='green')
            ax2.plot(df.index, df['Slope_slow'], label=f"{self.parameters['slow_window']}MA Slope", color='red')
            
            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Mark buy signals with 10MA slope
            for idx in buys.index:
                ax2.scatter(idx, df.loc[idx, 'Slope_mid'], marker='^', color='green', s=100)
            
            # Mark sell signals with 6MA slope
            for idx in sells.index:
                ax2.scatter(idx, df.loc[idx, 'Slope_fast'], marker='v', color='red', s=100)
            
            # Add threshold lines
            threshold = self.parameters['slope_threshold']
            ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label=f"Threshold: ±{threshold}")
            ax2.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
            
            ax2.set_title('Moving Average Slopes')
            ax2.set_ylabel('Slope Value')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save the figure
            chart_path = f"{debug_dir}/images/triple_ma_debug_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            print(f"Saved debug chart to {chart_path}")
        except Exception as e:
            print(f"ERROR in _generate_debug_charts: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def __str__(self):
        return (f"Triple MA Slope Strategy (Fast: {self.parameters['fast_window']}, "
                f"Mid: {self.parameters['mid_window']}, Slow: {self.parameters['slow_window']})") 