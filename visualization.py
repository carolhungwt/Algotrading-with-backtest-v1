import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import logging

class Visualizer:
    """Handles visualization of backtest results."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir (str, optional): Custom directory to save visualizations
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        self.output_dir = output_dir or 'output'
        self.images_dir = os.path.join(self.output_dir, 'images')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set_style("darkgrid")
    
    def plot_backtest_results(self, ticker, data, results):
        """
        Create and save visualizations of backtest results.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pandas.DataFrame): Historical price data
            results (dict): Dictionary containing backtest results
        """
        self.logger.info(f"Creating visualizations for {ticker}...")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create combined plot with three subplots
        self._plot_combined_results(ticker, data, results, timestamp)
        
        # Also create individual plots for more detail
        self._plot_portfolio_value(ticker, results, timestamp)
        self._plot_trading_signals(ticker, data, results, timestamp)
        self._plot_drawdown(ticker, results, timestamp)
        
        # Plot trade analysis
        if not results.get('trades', pd.DataFrame()).empty:
            self._plot_trade_analysis(ticker, results, timestamp)
    
    def _plot_combined_results(self, ticker, data, results, timestamp):
        """
        Create a single figure with three subplots: portfolio value, trading signals, and drawdown.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pandas.DataFrame): Historical price data
            results (dict): Dictionary containing backtest results
            timestamp (str): Timestamp for filename
        """
        portfolio_value = results.get('portfolio_value')
        trades = results.get('trades', pd.DataFrame())
        
        if portfolio_value is None:
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        
        # 1. Portfolio Value Plot
        axes[0].plot(portfolio_value.index, portfolio_value.values)
        axes[0].set_title(f'Portfolio Value - {ticker}')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)
        
        # Add annotations for metrics
        metrics_text = (
            f"Total Return: {results.get('total_return', 0):.2f}%\n"
            f"Annual Return: {results.get('annual_return', 0):.2f}%\n"
            f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%\n"
            f"Total Trades: {results.get('total_trades', 0)}"
        )
        axes[0].annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        # 2. Trading Signals Plot
        if not trades.empty:
            # Plot price
            axes[1].plot(data.index, data['Close'], label='Close Price', alpha=0.7)
            
            # Plot buy signals
            buy_trades = trades[trades['action'] == 'BUY']
            if not buy_trades.empty:
                buy_dates = pd.to_datetime(buy_trades['date'])
                buy_prices = buy_trades['price']
                axes[1].scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy')
            
            # Plot sell signals
            sell_trades = trades[trades['action'] == 'SELL']
            if not sell_trades.empty:
                sell_dates = pd.to_datetime(sell_trades['date'])
                sell_prices = sell_trades['price']
                axes[1].scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell')
            
            axes[1].set_title(f'Price Chart with Trading Signals - {ticker}')
            axes[1].set_ylabel('Price')
            axes[1].legend()
            axes[1].grid(True)
        
        # 3. Drawdown Plot
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value / running_max - 1) * 100
        
        axes[2].plot(drawdown.index, drawdown.values)
        axes[2].set_title(f'Portfolio Drawdown - {ticker}')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        axes[2].annotate(f'Max Drawdown: {max_dd:.2f}%', 
                     xy=(max_dd_date, max_dd),
                     xytext=(max_dd_date, max_dd - 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')
        
        # Format x-axis to show dates nicely
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_combined_results.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Combined results chart saved to {filename}")
    
    def _plot_portfolio_value(self, ticker, results, timestamp):
        """Plot portfolio value over time."""
        portfolio_value = results.get('portfolio_value')
        
        if portfolio_value is None:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value.values)
        plt.title(f'Portfolio Value - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Add annotations for metrics
        metrics_text = (
            f"Total Return: {results.get('total_return', 0):.2f}%\n"
            f"Annual Return: {results.get('annual_return', 0):.2f}%\n"
            f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%\n"
            f"Total Trades: {results.get('total_trades', 0)}"
        )
        plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_portfolio_value.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Portfolio value chart saved to {filename}")
    
    def _plot_trading_signals(self, ticker, data, results, timestamp):
        """Plot price chart with buy/sell signals."""
        trades = results.get('trades', pd.DataFrame())
        
        if trades.empty:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot price
        plt.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
        
        # Plot buy signals
        buy_trades = trades[trades['action'] == 'BUY']
        if not buy_trades.empty:
            buy_dates = pd.to_datetime(buy_trades['date'])
            buy_prices = buy_trades['price']
            plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy')
        
        # Plot sell signals
        sell_trades = trades[trades['action'] == 'SELL']
        if not sell_trades.empty:
            sell_dates = pd.to_datetime(sell_trades['date'])
            sell_prices = sell_trades['price']
            plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell')
        
        plt.title(f'Price Chart with Trading Signals - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_trading_signals.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Trading signals chart saved to {filename}")
    
    def _plot_drawdown(self, ticker, results, timestamp):
        """Plot drawdown over time."""
        portfolio_value = results.get('portfolio_value')
        
        if portfolio_value is None:
            return
            
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value / running_max - 1) * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values)
        plt.title(f'Portfolio Drawdown - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        plt.annotate(f'Max Drawdown: {max_dd:.2f}%', 
                     xy=(max_dd_date, max_dd),
                     xytext=(max_dd_date, max_dd - 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_drawdown.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Drawdown chart saved to {filename}")
    
    def _plot_trade_analysis(self, ticker, results, timestamp):
        """Plot trade analysis charts."""
        trades = results.get('trades', pd.DataFrame())
        
        if trades.empty or 'profit_loss' not in trades.columns:
            return
            
        # Filter to only include sell trades with P&L info
        sell_trades = trades[trades['action'] == 'SELL'].dropna(subset=['profit_loss'])
        
        if sell_trades.empty:
            return
            
        # 1. Profit/Loss per trade
        plt.figure(figsize=(12, 6))
        
        # Convert dates to datetime if they're not already
        sell_trades['date'] = pd.to_datetime(sell_trades['date'])
        
        # Sort by date
        sell_trades = sell_trades.sort_values('date')
        
        # Plot P&L
        colors = ['green' if x > 0 else 'red' for x in sell_trades['profit_loss']]
        plt.bar(range(len(sell_trades)), sell_trades['profit_loss'], color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title(f'Profit/Loss per Trade - {ticker}')
        plt.xlabel('Trade Number')
        plt.ylabel('Profit/Loss ($)')
        plt.grid(True, axis='y')
        
        # Add cumulative P&L
        cum_pnl = sell_trades['profit_loss'].cumsum()
        ax2 = plt.twinx()
        ax2.plot(range(len(sell_trades)), cum_pnl, color='blue', linestyle='-', marker='o')
        ax2.set_ylabel('Cumulative P&L ($)', color='blue')
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_trade_pnl.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Trade P&L chart saved to {filename}")
        
        # 2. Trade duration analysis
        if 'trade_id' in sell_trades.columns and 'trade_id' in trades.columns:
            plt.figure(figsize=(12, 6))
            
            # Calculate trade durations
            durations = []
            returns = []
            
            for trade_id in sell_trades['trade_id'].unique():
                buy_date = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'BUY')]['date'].values
                sell_date = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'SELL')]['date'].values
                
                if len(buy_date) > 0 and len(sell_date) > 0:
                    buy_date = pd.to_datetime(buy_date[0])
                    sell_date = pd.to_datetime(sell_date[0])
                    duration = (sell_date - buy_date).days
                    durations.append(duration)
                    
                    ret = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'SELL')]['profit_loss_pct'].values
                    if len(ret) > 0:
                        returns.append(ret[0])
            
            if durations and returns:
                # Create scatter plot of duration vs return
                plt.scatter(durations, returns, alpha=0.7)
                
                # Add trend line
                if len(durations) > 1:
                    z = np.polyfit(durations, returns, 1)
                    p = np.poly1d(z)
                    plt.plot(durations, p(durations), "r--", alpha=0.7)
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f'Trade Duration vs Return - {ticker}')
                plt.xlabel('Trade Duration (days)')
                plt.ylabel('Return (%)')
                plt.grid(True)
                
                # Save the figure
                filename = os.path.join(self.images_dir, f'{ticker}_trade_duration.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Trade duration analysis saved to {filename}")
    
    def plot_comparison(self, results_dict, benchmark_data=None):
        """
        Plot comparison of multiple backtest results.
        
        Args:
            results_dict (dict): Dictionary mapping strategy names to backtest results
            benchmark_data (pandas.DataFrame, optional): Benchmark data for comparison
        """
        if not results_dict:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio values for each strategy
        for strategy_name, results in results_dict.items():
            portfolio_value = results.get('portfolio_value')
            if portfolio_value is not None:
                # Normalize to percentage change from start
                normalized = (portfolio_value / portfolio_value.iloc[0] - 1) * 100
                plt.plot(normalized.index, normalized.values, label=strategy_name)
        
        # Plot benchmark if provided
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_close = benchmark_data['Close']
            normalized_benchmark = (benchmark_close / benchmark_close.iloc[0] - 1) * 100
            plt.plot(normalized_benchmark.index, normalized_benchmark.values, 
                     label='Benchmark', color='black', linestyle='--')
        
        plt.title('Strategy Comparison')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        filename = os.path.join(self.images_dir, 'strategy_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Strategy comparison chart saved to {filename}")
    
    def generate_performance_report(self, ticker, results, output_format='html'):
        """
        Generate a comprehensive performance report.
        
        Args:
            ticker (str): Stock ticker symbol
            results (dict): Dictionary containing backtest results
            output_format (str): Output format ('html', 'pdf')
            
        Returns:
            str: Path to the generated report
        """
        # This would be an advanced feature to implement
        # Would require additional libraries like Jinja2 for templating
        # or ReportLab for PDF generation
        
        self.logger.info(f"Performance report generation not implemented yet")
        return None 