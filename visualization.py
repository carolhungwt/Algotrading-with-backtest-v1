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
    
    def __init__(self, output_dir=None, scanning_mode=False, scan_for=None):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir (str, optional): Custom directory to save visualizations
            scanning_mode (bool): Whether this is for signal scanning mode
            scan_for (str, optional): Type of signal being scanned ('buy' or 'sell')
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Scanning mode settings
        self.scanning_mode = scanning_mode
        self.scan_for = scan_for.lower() if scan_for else None
        
        # Set output directory based on mode
        if scanning_mode:
            # Use the backtest scanning output directory structure
            base_dir = 'backtest_scanning_output'
            scan_type_dir = f"{self.scan_for}_signals" if self.scan_for else 'signals'
            self.output_dir = output_dir or os.path.join(base_dir, scan_type_dir)
        else:
            # Regular backtest mode
            self.output_dir = output_dir or 'output'
            
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.csv_dir = os.path.join(self.output_dir, 'csv')
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.images_dir, self.csv_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
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
            
            # Plot stop loss levels if available
            if 'stop_loss_price' in trades.columns:
                stop_loss_trades = trades[(trades['action'] == 'BUY') & (~trades['stop_loss_price'].isna())]
                if not stop_loss_trades.empty:
                    for _, trade in stop_loss_trades.iterrows():
                        trade_id = trade['trade_id']
                        buy_date = pd.to_datetime(trade['date'])
                        stop_price = trade['stop_loss_price']
                        
                        # Find the sell date for this trade
                        sell_trade = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'SELL')]
                        if not sell_trade.empty:
                            sell_date = pd.to_datetime(sell_trade['date'].values[0])
                            
                            # Plot horizontal line for stop loss level from buy date to sell date
                            date_range = pd.date_range(start=buy_date, end=sell_date)
                            if len(date_range) > 0:
                                axes[1].plot(date_range, [stop_price] * len(date_range), 
                                         'r--', alpha=0.6, linewidth=1)
            
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
        
        # Plot stop loss levels if available
        if 'stop_loss_price' in trades.columns:
            stop_loss_trades = trades[(trades['action'] == 'BUY') & (~trades['stop_loss_price'].isna())]
            if not stop_loss_trades.empty:
                for _, trade in stop_loss_trades.iterrows():
                    trade_id = trade['trade_id']
                    buy_date = pd.to_datetime(trade['date'])
                    stop_price = trade['stop_loss_price']
                    
                    # Find the sell date for this trade
                    sell_trade = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'SELL')]
                    if not sell_trade.empty:
                        sell_date = pd.to_datetime(sell_trade['date'].values[0])
                        
                        # Plot horizontal line for stop loss level from buy date to sell date
                        date_range = pd.date_range(start=buy_date, end=sell_date)
                        if len(date_range) > 0:
                            plt.plot(date_range, [stop_price] * len(date_range), 
                                     'r--', alpha=0.6, linewidth=1)
                            
                            # Add a small text label
                            plt.annotate('Stop', xy=(buy_date, stop_price),
                                         xytext=(buy_date, stop_price * 0.98),
                                         color='red', alpha=0.7, fontsize=8)
        
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
    
    def plot_scanning_results(self, ticker, data, results):
        """
        Create and save visualizations of signal scanning results.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pandas.DataFrame): Historical price data
            results (dict): Dictionary containing scanning results
        """
        self.logger.info(f"Creating signal scanning visualizations for {ticker}...")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create overview plot
        self._plot_signal_overview(ticker, data, results, timestamp)
        
        # Create signal accuracy analysis
        if 'signals' in results and not results['signals'].empty:
            self._plot_signal_accuracy(ticker, results, timestamp)
            
            # Save signals data to CSV
            self._save_signals_data(ticker, results, timestamp)
        
        # Save full signals data with price
        if 'signals_data' in results and not results['signals_data'].empty:
            self._save_full_signals_data(ticker, results, timestamp)
    
    def _plot_signal_overview(self, ticker, data, results, timestamp):
        """
        Create an overview plot of price with signals.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pandas.DataFrame): Historical price data
            results (dict): Dictionary containing scanning results
            timestamp (str): Timestamp for filename
        """
        signals_data = results.get('signals_data')
        scan_for = results.get('scan_for', 'buy')
        signals_df = results.get('signals', pd.DataFrame())
        
        if signals_data is None:
            return
            
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on the top subplot
        axes[0].plot(data.index, data['Close'], label='Close Price', alpha=0.7)
        
        # Add SMA
        if len(data) > 20:
            sma20 = data['Close'].rolling(window=20).mean()
            axes[0].plot(data.index, sma20, label='20-day SMA', color='purple', 
                      linestyle='--', alpha=0.6)
            
        # Plot signals if available
        if not signals_df.empty:
            signal_dates = pd.to_datetime(signals_df['date'])
            signal_prices = signals_df['price']
            
            marker = '^' if scan_for == 'buy' else 'v'
            color = 'green' if scan_for == 'buy' else 'red'
            label = f"{scan_for.capitalize()} Signal"
            
            axes[0].scatter(signal_dates, signal_prices, marker=marker, color=color, 
                         s=100, label=label)
            
            # Annotate signal strength for a subset of signals
            if len(signal_dates) <= 10:
                # If few signals, annotate all
                for i, (date, price, strength) in enumerate(
                    zip(signal_dates, signal_prices, signals_df['strength'])):
                    axes[0].annotate(
                        f"{strength:.2f}", 
                        (date, price),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
        
        # Bottom subplot: Signal strength or distribution
        if 'signal' in signals_data.columns:
            # For buy signals, focus on positive values
            if scan_for == 'buy':
                signal_values = signals_data['signal'].apply(lambda x: max(0, x))
                threshold = results.get('signal_threshold', 0.2)
            # For sell signals, focus on negative values (convert to positive for display)
            else:
                signal_values = signals_data['signal'].apply(lambda x: max(0, -x))
                threshold = results.get('signal_threshold', 0.2)
            
            # Plot signal strength
            axes[1].plot(data.index, signal_values, color='orange', label='Signal Strength')
            axes[1].axhline(y=threshold, color='red', linestyle='--', 
                         label=f'Signal Threshold ({threshold})')
            
            axes[1].set_title(f"{scan_for.capitalize()} Signal Strength")
            axes[1].set_ylabel('Strength')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Set titles and labels
        axes[0].set_title(f"{scan_for.capitalize()} Signal Scan - {ticker}")
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.images_dir, f'{ticker}_{scan_for}_signal_scan.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Signal scan chart saved to {filename}")
    
    def _plot_signal_accuracy(self, ticker, results, timestamp):
        """
        Plot signal accuracy analysis.
        
        Args:
            ticker (str): Stock ticker symbol
            results (dict): Dictionary containing scanning results
            timestamp (str): Timestamp for filename
        """
        signals_df = results.get('signals', pd.DataFrame())
        scan_for = results.get('scan_for', 'buy')
        
        if signals_df.empty:
            return
            
        # Create figure for accuracy metrics
        plt.figure(figsize=(12, 8))
        
        # Bar chart for signal accuracy at different time horizons
        accuracy_columns = [col for col in signals_df.columns if col.startswith('accuracy_')]
        if accuracy_columns:
            accuracies = []
            periods = []
            
            for col in sorted(accuracy_columns, key=lambda x: int(x.split('_')[1][:-1])):
                # Extract period (e.g., "accuracy_5d" -> 5)
                period = int(col.split('_')[1][:-1])
                periods.append(f"{period}d")
                
                # Get accuracy value (all rows should have the same value)
                accuracy = signals_df[col].iloc[0]
                accuracies.append(accuracy)
            
            # Plot bar chart
            bars = plt.bar(periods, accuracies, alpha=0.7, 
                         color='green' if scan_for == 'buy' else 'red')
            
            # Add value labels on top of each bar
            for bar, accuracy in zip(bars, accuracies):
                plt.annotate(f"{accuracy:.1f}%",
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, 
                      label='Random Guess (50%)')
            
            title_prefix = "Buy" if scan_for == 'buy' else "Sell"
            plt.title(f"{title_prefix} Signal Accuracy - {ticker}")
            plt.xlabel('Time Horizon')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            plt.grid(True, axis='y')
            plt.legend()
            
            # Save the figure
            filename = os.path.join(self.images_dir, f'{ticker}_{scan_for}_signal_accuracy.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Signal accuracy chart saved to {filename}")
            
        # Create figure for price change distribution
        plt.figure(figsize=(12, 8))
        
        # Histogram of price changes after signals
        price_change_columns = [col for col in signals_df.columns 
                              if col.startswith('price_change_') and col.endswith('_pct')]
        
        if price_change_columns and not signals_df.empty:
            for col in sorted(price_change_columns, 
                            key=lambda x: int(x.split('_')[2][:-1])):
                # Extract period (e.g., "price_change_5d_pct" -> 5)
                period = int(col.split('_')[2][:-1])
                
                # Create histogram
                plt.hist(signals_df[col], bins=20, alpha=0.5, 
                       label=f"{period}-day Change")
            
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            
            title_prefix = "After Buy" if scan_for == 'buy' else "After Sell"
            plt.title(f"Price Change Distribution {title_prefix} Signals - {ticker}")
            plt.xlabel('Price Change (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.legend()
            
            # Save the figure
            filename = os.path.join(self.images_dir, 
                                  f'{ticker}_{scan_for}_price_distribution.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Price distribution chart saved to {filename}")
    
    def _save_signals_data(self, ticker, results, timestamp):
        """
        Save signal data to CSV.
        
        Args:
            ticker (str): Stock ticker symbol
            results (dict): Dictionary containing scanning results
            timestamp (str): Timestamp for filename
        """
        signals_df = results.get('signals', pd.DataFrame())
        scan_for = results.get('scan_for', 'buy')
        
        if signals_df.empty:
            return
            
        # Save to CSV
        filename = os.path.join(self.csv_dir, f'{ticker}_{scan_for}_signals.csv')
        signals_df.to_csv(filename, index=False)
        self.logger.info(f"Signal data saved to {filename}")
    
    def _save_full_signals_data(self, ticker, results, timestamp):
        """
        Save full price and signal data to CSV.
        
        Args:
            ticker (str): Stock ticker symbol
            results (dict): Dictionary containing scanning results
            timestamp (str): Timestamp for filename
        """
        signals_data = results.get('signals_data')
        scan_for = results.get('scan_for', 'buy')
        
        if signals_data is None:
            return
            
        # Save to CSV
        filename = os.path.join(self.csv_dir, f'{ticker}_{scan_for}_full_data.csv')
        signals_data.to_csv(filename)
        self.logger.info(f"Full signal data saved to {filename}") 