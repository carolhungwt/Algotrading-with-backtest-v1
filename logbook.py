import pandas as pd
import os
from datetime import datetime
import json

class Logbook:
    """Tracks and logs strategy performance across multiple backtests."""
    
    def __init__(self, log_dir="logs"):
        """
        Initialize the logbook system.
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create ticker-specific directories
        self.ticker_dir = os.path.join(log_dir, "tickers")
        if not os.path.exists(self.ticker_dir):
            os.makedirs(self.ticker_dir)
            
        # Initialize master log file if it doesn't exist
        self.master_log_path = os.path.join(log_dir, "master_log.csv")
        if not os.path.exists(self.master_log_path):
            # Create empty DataFrame with desired columns
            columns = [
                'timestamp', 'ticker', 'buy_strategy', 'sell_strategy', 
                'use_stop_loss', 'stop_loss_settings', 'initial_capital',
                'net_profit', 'total_return', 'duration', 'interval',
                'num_trades', 'win_rate', 'backtest_id'
            ]
            master_df = pd.DataFrame(columns=columns)
            master_df.to_csv(self.master_log_path, index=False)
    
    def log_backtest(self, ticker, backtest_results, strategy_info, args, backtest_id):
        """
        Log a completed backtest result.
        
        Args:
            ticker (str): Stock ticker symbol
            backtest_results (dict): Results from the backtest
            strategy_info (dict): Information about strategies used
            args (argparse.Namespace): Arguments used for the backtest
            backtest_id (str): Unique identifier for this backtest run
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create basic log entry
        log_entry = {
            'timestamp': timestamp,
            'ticker': ticker,
            'backtest_id': backtest_id,
            'initial_capital': args.initial_capital,
            'interval': args.interval,
            'duration': args.period,
            'net_profit': backtest_results['final_value'] - args.initial_capital,
            'total_return': backtest_results['total_return'],
            'annual_return': backtest_results['annual_return'],
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'max_drawdown': backtest_results['max_drawdown'],
            'num_trades': backtest_results['total_trades'],
            'win_rate': backtest_results['win_rate'],
            'commission_rate': args.commission
        }
        
        # Add strategy information
        if args.separate_signals:
            log_entry['buy_strategy'] = strategy_info.get('buy_strategies', '')
            log_entry['sell_strategy'] = strategy_info.get('sell_strategies', '')
        else:
            log_entry['buy_strategy'] = strategy_info.get('strategies', '')
            log_entry['sell_strategy'] = strategy_info.get('strategies', '')
            
        # Add signal combination method and threshold
        log_entry['combine_method'] = args.combine_method
        log_entry['signal_threshold'] = args.signal_threshold
        
        # Add stop loss information
        log_entry['use_stop_loss'] = args.use_stop_loss
        
        if args.use_stop_loss:
            stop_loss_settings = {
                'atr_multiplier': args.stop_loss_atr_multiplier,
                'atr_period': args.stop_loss_atr_period
            }
            log_entry['stop_loss_settings'] = json.dumps(stop_loss_settings)
            
            # Add stop loss performance metrics if available
            for metric in ['stopped_out_count', 'stopped_out_pl', 'stopped_win_rate', 
                          'normal_pl', 'normal_win_rate']:
                if metric in backtest_results:
                    log_entry[metric] = backtest_results[metric]
        else:
            log_entry['stop_loss_settings'] = '{}'
            
        # Update master log
        self._update_master_log(log_entry)
        
        # Update ticker-specific log
        self._update_ticker_log(ticker, log_entry)
        
        # Log detailed trade information
        if 'trades' in backtest_results and not backtest_results['trades'].empty:
            self._log_trade_details(ticker, backtest_results['trades'], backtest_id)
            
    def _update_master_log(self, log_entry):
        """Update the master log file with a new entry."""
        try:
            # Load existing log
            master_df = pd.read_csv(self.master_log_path)
            
            # Append new entry
            new_entry_df = pd.DataFrame([log_entry])
            master_df = pd.concat([master_df, new_entry_df], ignore_index=True)
            
            # Save updated log
            master_df.to_csv(self.master_log_path, index=False)
        except Exception as e:
            print(f"Error updating master log: {e}")
            
    def _update_ticker_log(self, ticker, log_entry):
        """Update the ticker-specific log file."""
        ticker_log_path = os.path.join(self.ticker_dir, f"{ticker}_log.csv")
        
        try:
            # Load existing ticker log or create new one
            if os.path.exists(ticker_log_path):
                ticker_df = pd.read_csv(ticker_log_path)
                new_entry_df = pd.DataFrame([log_entry])
                ticker_df = pd.concat([ticker_df, new_entry_df], ignore_index=True)
            else:
                ticker_df = pd.DataFrame([log_entry])
                
            # Save updated log
            ticker_df.to_csv(ticker_log_path, index=False)
        except Exception as e:
            print(f"Error updating ticker log for {ticker}: {e}")
            
    def _log_trade_details(self, ticker, trades_df, backtest_id):
        """Log detailed information about individual trades."""
        # Create trades directory if it doesn't exist
        trades_dir = os.path.join(self.log_dir, "trades")
        if not os.path.exists(trades_dir):
            os.makedirs(trades_dir)
            
        # Add backtest_id to trades dataframe for reference
        trades_df = trades_df.copy()
        trades_df['backtest_id'] = backtest_id
        trades_df['ticker'] = ticker
        
        # Save trade details
        trade_log_path = os.path.join(trades_dir, f"{ticker}_{backtest_id}_trades.csv")
        trades_df.to_csv(trade_log_path, index=False)
        
    def get_ticker_performance(self, ticker):
        """
        Get performance history for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pandas.DataFrame: Performance history for the ticker
        """
        ticker_log_path = os.path.join(self.ticker_dir, f"{ticker}_log.csv")
        
        if os.path.exists(ticker_log_path):
            return pd.read_csv(ticker_log_path)
        else:
            return pd.DataFrame()
            
    def get_strategy_performance(self, strategy_name):
        """
        Get performance of a specific strategy across all tickers.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            pandas.DataFrame: Performance of the strategy
        """
        try:
            master_df = pd.read_csv(self.master_log_path)
            
            # Filter for the strategy in either buy or sell strategies
            strategy_df = master_df[
                (master_df['buy_strategy'].str.contains(strategy_name, na=False)) | 
                (master_df['sell_strategy'].str.contains(strategy_name, na=False))
            ]
            
            return strategy_df
        except Exception:
            return pd.DataFrame()
            
    def get_best_strategies(self, ticker=None, metric='total_return', top_n=5):
        """
        Get the best performing strategies based on a metric.
        
        Args:
            ticker (str, optional): Filter by ticker
            metric (str): Metric to use for ranking ('total_return', 'sharpe_ratio', etc.)
            top_n (int): Number of top strategies to return
            
        Returns:
            pandas.DataFrame: Top performing strategies
        """
        try:
            master_df = pd.read_csv(self.master_log_path)
            
            if ticker:
                master_df = master_df[master_df['ticker'] == ticker]
                
            if len(master_df) == 0:
                return pd.DataFrame()
                
            # Sort by the specified metric
            if metric in master_df.columns:
                sorted_df = master_df.sort_values(by=metric, ascending=False)
                return sorted_df.head(top_n)
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame() 