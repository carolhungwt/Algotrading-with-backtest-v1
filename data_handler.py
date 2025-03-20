import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

class DataHandler:
    """
    Handles fetching, processing and storing of stock market data.
    """
    
    def __init__(self):
        """Initialize the DataHandler."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, ticker, period='1y', interval='1d'):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Data period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            pandas.DataFrame: Stock data with columns [Open, High, Low, Close, Volume]
        """
        try:
            self.logger.info(f"Fetching data for {ticker} with period={period}, interval={interval}")
            
            # Attempt to fetch data with exponential backoff for retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(ticker, period=period, interval=interval, progress=False)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Error fetching data, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if data.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return None
            
            # Process the data
            data = self._process_data(data)
            
            self.logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def _process_data(self, data):
        """
        Process the raw data from Yahoo Finance using the stack approach.
        
        Args:
            data (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Processed stock data
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = data.copy()
            
            # Check if we're dealing with a MultiIndex dataframe
            if isinstance(df.columns, pd.MultiIndex):
                # Use the stack approach to reshape the data
                stacked = df.stack(level=0).reset_index()
                
                # Rename columns appropriately
                stacked.columns = ['Date', 'Symbol', 'Value']
                
                # Pivot the data back into the format we need
                df = stacked.pivot(index='Date', columns='Symbol')
                
                # Flatten the MultiIndex columns
                df.columns = df.columns.get_level_values(1)
            
            # Standardize column names (capitalize first letter)
            df.columns = [col.capitalize() for col in df.columns]
            
            # Standard columns we expect in OHLCV data
            standard_columns = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj close': 'Close',
                'adj. close': 'Close',
                'adjusted close': 'Close'
            }
            
            # Create a mapping for column renaming
            rename_map = {}
            for col in df.columns:
                lower_col = col.lower()
                if lower_col in standard_columns:
                    rename_map[col] = standard_columns[lower_col]
            
            # Apply the renaming if we found any matches
            if rename_map:
                df = df.rename(columns=rename_map)
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.warning(f"Missing column {col} in data")
            
            # Drop rows with NaN values in the Close price
            if 'Close' in df.columns:
                df = df.dropna(subset=['Close'])
                
                # Add returns column
                df['Returns'] = df['Close'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            # If there's an error, return the original dataframe to avoid breaking things
            return data
    
    def get_multiple_stocks_data(self, tickers, period='1y', interval='1d'):
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers (list): List of stock ticker symbols
            period (str): Data period
            interval (str): Data interval
            
        Returns:
            dict: Dictionary mapping ticker symbols to their respective data frames
        """
        result = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period, interval)
            if data is not None:
                result[ticker] = data
        return result
    
    def save_data_to_csv(self, data, filename):
        """
        Save stock data to a CSV file.
        
        Args:
            data (pandas.DataFrame): Stock data
            filename (str): Output filename
        """
        try:
            data.to_csv(filename)
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {str(e)}") 