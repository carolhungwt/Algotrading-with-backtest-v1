import argparse
import pandas as pd
from datetime import datetime
import os
import json
import shutil
import numpy as np
import uuid

from data_handler import DataHandler
from backtest_engine import BacktestEngine
from strategy_manager import StrategyManager
from visualization import Visualizer
from logbook import Logbook

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists("output"):
        os.makedirs("output")

def create_backtest_directory():
    """Create a unique directory for this backtest run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_id = f"backtest_{timestamp}"
    backtest_dir = f"output/{backtest_id}"
    os.makedirs(backtest_dir)
    os.makedirs(f"{backtest_dir}/images")
    os.makedirs(f"{backtest_dir}/csv")
    return backtest_dir, backtest_id

def save_backtest_summary(backtest_dir, args, tickers, strategy_info, results_summary):
    """Save a summary text file with all backtest parameters and results"""
    summary_path = f"{backtest_dir}/summary.txt"
    
    with open(summary_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"BACKTEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Write date and time
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write data parameters
        f.write("DATA PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Tickers: {', '.join(tickers)}\n")
        f.write(f"Period: {args.period}\n")
        f.write(f"Interval: {args.interval}\n\n")
        
        # Write strategy parameters
        f.write("STRATEGY PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        
        if args.separate_signals:
            f.write("Using separate strategies for buy and sell signals\n")
            f.write(f"Buy strategies: {strategy_info['buy_strategies']}\n")
            f.write(f"Sell strategies: {strategy_info['sell_strategies']}\n")
        else:
            f.write(f"Strategies: {strategy_info['strategies']}\n")
        
        f.write(f"Signal combination method: {args.combine_method}\n")
        f.write(f"Signal threshold: {args.signal_threshold}\n")
        
        # Write stop loss parameters if enabled
        if args.use_stop_loss:
            f.write(f"Stop Loss: Enabled\n")
            f.write(f"Stop Loss ATR Multiplier: {args.stop_loss_atr_multiplier}\n")
            f.write(f"Stop Loss ATR Period: {args.stop_loss_atr_period}\n")
        else:
            f.write(f"Stop Loss: Disabled\n")
        
        # Write custom parameters if provided
        if args.params and args.params.strip():
            f.write(f"Custom parameters: {args.params}\n")
        
        f.write("\n")
        
        # Write backtest parameters
        f.write("BACKTEST PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial capital: ${args.initial_capital:.2f}\n")
        f.write(f"Commission rate: {args.commission * 100:.3f}%\n\n")
        
        # Write results for each ticker
        f.write("RESULTS:\n")
        f.write("-" * 80 + "\n")
        
        for ticker, results in results_summary.items():
            f.write(f"\nResults for {ticker}:\n")
            f.write(f"  Final Portfolio Value: ${results['final_value']:.2f}\n")
            f.write(f"  Net Profit: ${results['final_value'] - args.initial_capital:.2f}\n")
            f.write(f"  Total Return: {results['total_return']:.2f}%\n")
            f.write(f"  Annual Return: {results['annual_return']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown: {results['max_drawdown']:.2f}%\n")
            f.write(f"  Number of Trades: {results['total_trades']}\n")
            f.write(f"  Win Rate: {results['win_rate']:.2f}%\n")
            
            # Calculate commission costs if trades were made
            if 'commission_total' in results:
                f.write(f"  Total Commission Costs: ${results['commission_total']:.2f}\n")
            
            # Add stop loss information to each ticker's results
            if args.use_stop_loss:
                # Stop loss configuration
                f.write(f"\n  STOP LOSS DETAILS (ATR × {args.stop_loss_atr_multiplier}, {args.stop_loss_atr_period} periods):\n")
                
                # Calculate how many trades were stopped out by stop loss, if available
                if 'stopped_out_count' in results:
                    stop_percentage = results['stopped_out_count'] / max(results['total_trades'], 1) * 100
                    f.write(f"    Trades Stopped Out: {results['stopped_out_count']} ({stop_percentage:.1f}% of trades)\n")
                
                # Include profit/loss comparisons if available
                if 'stopped_out_pl' in results and 'normal_pl' in results:
                    f.write(f"    Stopped Out Trades P&L: ${results['stopped_out_pl']:.2f} (Avg: ${results['avg_stopped_pl']:.2f})\n")
                    f.write(f"    Normal Exit Trades P&L: ${results['normal_pl']:.2f} (Avg: ${results['avg_normal_pl']:.2f})\n")
                    f.write(f"    Stopped Out Win Rate: {results['stopped_win_rate']:.1f}%\n")
                    f.write(f"    Normal Exit Win Rate: {results['normal_win_rate']:.1f}%\n")
            
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='AlgoTrade Backtesting System')
    
    # Stock and data parameters
    parser.add_argument('--tickers', type=str, required=True, help='Comma-separated list of stock tickers')
    parser.add_argument('--period', type=str, default='1y', help='Period of historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')
    
    # Strategy parameters - with new options for separate buy/sell strategies
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument('--strategies', type=str, help='Comma-separated list of strategies to use for both buy and sell signals')
    strategy_group.add_argument('--separate-signals', action='store_true', help='Use separate strategies for buy and sell signals')
    
    parser.add_argument('--buy-strategies', type=str, help='Comma-separated list of strategies to use for buy signals only')
    parser.add_argument('--sell-strategies', type=str, help='Comma-separated list of strategies to use for sell signals only')
    parser.add_argument('--params', type=str, default='', help='JSON string of strategy parameters')
    
    # Signal combination parameters
    parser.add_argument('--combine-method', type=str, default='average', 
                       choices=['average', 'vote', 'unanimous', 'majority', 'weighted'],
                       help='Method to combine signals from multiple strategies')
    parser.add_argument('--signal-threshold', type=float, default=0.2, 
                       help='Threshold for converting continuous signals to discrete buy/sell signals')
    
    # Stop Loss parameters
    parser.add_argument('--use-stop-loss', action='store_true', 
                       help='Enable stop loss orders based on ATR')
    parser.add_argument('--stop-loss-atr-multiplier', type=float, default=1.0, 
                       help='Multiplier for ATR to determine stop loss distance (default: 1.0)')
    parser.add_argument('--stop-loss-atr-period', type=int, default=14, 
                       help='Period for ATR calculation (default: 14)')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital for backtesting')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate per trade')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--list-strategies', action='store_true', help='List all available strategies and exit')
    
    # New logging specific arguments
    parser.add_argument('--no-logging', action='store_true', help='Disable logging to the logbook system')
    
    args = parser.parse_args()
    
    # Create base output directory
    create_output_directory()
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    
    # Initialize logbook
    logbook = Logbook()
    
    # If list_strategies flag is set, print available strategies and exit
    if args.list_strategies:
        available_strategies = strategy_manager.get_available_strategies()
        print("Available strategies:")
        for strategy in available_strategies:
            print(f"  - {strategy}")
        return
    
    # Parse parameters if provided
    params = {}
    if args.params and args.params.strip():
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing strategy parameters: {str(e)}")
            return
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Initialize components
    data_handler = DataHandler()
    
    # Create a unique directory for this backtest run
    backtest_dir, backtest_id = create_backtest_directory()
    print(f"Saving backtest results to: {backtest_dir}")
    
    # Create visualizer with custom output directory
    visualizer = Visualizer(output_dir=backtest_dir)
    
    # Load strategies based on whether using separate signals
    strategies = []
    buy_strategies = []
    sell_strategies = []
    strategy_info = {'strategies': '', 'buy_strategies': '', 'sell_strategies': ''}
    
    if args.separate_signals:
        if not args.buy_strategies or not args.sell_strategies:
            print("Error: When using --separate-signals, both --buy-strategies and --sell-strategies must be provided")
            return
            
        # Load buy strategies
        buy_strategy_names = [s.strip() for s in args.buy_strategies.split(',')]
        buy_strategies = strategy_manager.load_strategies(buy_strategy_names, args.params)
        
        # Load sell strategies
        sell_strategy_names = [s.strip() for s in args.sell_strategies.split(',')]
        sell_strategies = strategy_manager.load_strategies(sell_strategy_names, args.params)
        
        strategy_info['buy_strategies'] = ', '.join(buy_strategy_names)
        strategy_info['sell_strategies'] = ', '.join(sell_strategy_names)
        
        print(f"Starting backtest for tickers: {', '.join(tickers)}")
        print(f"Using separate strategies:")
        print(f"  Buy strategies: {', '.join(buy_strategy_names)}")
        print(f"  Sell strategies: {', '.join(sell_strategy_names)}")
    else:
        # Use same strategies for both buy and sell
        strategy_names = [s.strip() for s in args.strategies.split(',')]
        strategies = strategy_manager.load_strategies(strategy_names, args.params)
        
        strategy_info['strategies'] = ', '.join(strategy_names)
        
        print(f"Starting backtest for tickers: {', '.join(tickers)}")
        print(f"Using strategies: {', '.join(strategy_names)}")
    
    print(f"Signal combination method: {args.combine_method}")
    
    # Print stop loss details if enabled
    if args.use_stop_loss:
        print(f"Stop loss enabled: ATR multiplier = {args.stop_loss_atr_multiplier}, ATR period = {args.stop_loss_atr_period}")
    
    # Dictionary to store results for summary
    results_summary = {}
    
    # Run backtest for each ticker
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Get data
        data = data_handler.get_stock_data(ticker, args.period, args.interval)
        if data is None or len(data) == 0:
            print(f"No data available for {ticker}, skipping...")
            continue
            
        # Initialize backtest engine with appropriate parameters
        if args.separate_signals:
            backtest = BacktestEngine(
                data=data,
                buy_strategies=buy_strategies,
                sell_strategies=sell_strategies,
                initial_capital=args.initial_capital,
                commission=args.commission,
                signal_threshold=args.signal_threshold,
                combine_method=args.combine_method,
                separate_signals=True,
                use_stop_loss=args.use_stop_loss,
                stop_loss_atr_multiplier=args.stop_loss_atr_multiplier,
                stop_loss_atr_period=args.stop_loss_atr_period
            )
        else:
            backtest = BacktestEngine(
                data=data,
                strategies=strategies,
                initial_capital=args.initial_capital,
                commission=args.commission,
                signal_threshold=args.signal_threshold,
                combine_method=args.combine_method,
                separate_signals=False,
                use_stop_loss=args.use_stop_loss,
                stop_loss_atr_multiplier=args.stop_loss_atr_multiplier,
                stop_loss_atr_period=args.stop_loss_atr_period
            )
        
        # Run backtest
        results = backtest.run()
        
        # Store results for summary
        results_summary[ticker] = {
            'final_value': results['final_value'],
            'total_return': results['total_return'],
            'annual_return': results['annual_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades']
        }
        
        # Include stop loss metrics if available
        if args.use_stop_loss:
            stop_loss_metrics = [
                'stopped_out_count', 'stop_loss_atr_multiplier', 'stop_loss_atr_period',
                'stopped_out_pl', 'avg_stopped_pl', 'stopped_win_rate',
                'normal_pl', 'avg_normal_pl', 'normal_win_rate'
            ]
            
            for metric in stop_loss_metrics:
                if metric in results:
                    results_summary[ticker][metric] = results[metric]
        
        # Calculate total commission if trades were made
        if not results['trades'].empty:
            total_commission = results['trades']['commission'].sum()
            results_summary[ticker]['commission_total'] = total_commission
        
        # Display results
        print("\nBacktest Results:")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Number of Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        
        # Log results to the logbook system
        if not args.no_logging:
            logbook.log_backtest(
                ticker=ticker,
                backtest_results=results,
                strategy_info=strategy_info,
                args=args,
                backtest_id=backtest_id
            )
            print(f"Results logged to logbook for {ticker}")
        
        # Visualize results to the backtest directory
        visualizer.plot_backtest_results(ticker, data, results)
        
        # Save trades to CSV in the backtest directory
        if not results['trades'].empty:
            # Process the trades dataframe to enhance stop loss information
            trades_df = results['trades'].copy()
            
            # Add a more descriptive stop loss column
            if 'stopped_out' in trades_df.columns:
                # Initialize with object dtype to allow string values
                trades_df['Stop Loss Triggered'] = pd.NA
                # Then map the values
                trades_df.loc[trades_df['stopped_out'] == True, 'Stop Loss Triggered'] = 'Yes'
                trades_df.loc[trades_df['stopped_out'] == False, 'Stop Loss Triggered'] = 'No'
                trades_df.loc[trades_df['stopped_out'].isna(), 'Stop Loss Triggered'] = 'N/A'
            
            # Rename columns for clarity
            trades_df = trades_df.rename(columns={
                'date': 'Date', 
                'action': 'Action',
                'price': 'Price',
                'shares': 'Shares',
                'value': 'Value',
                'commission': 'Commission',
                'trade_id': 'Trade ID',
                'stop_loss_price': 'Stop Loss Price',
                'profit_loss': 'Profit/Loss ($)',
                'profit_loss_pct': 'Profit/Loss (%)'
            })
            
            # Format numeric columns to be more readable
            for col in ['Price', 'Value', 'Commission', 'Stop Loss Price']:
                if col in trades_df.columns:
                    trades_df[col] = trades_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
            
            # Format profit/loss columns
            if 'Profit/Loss ($)' in trades_df.columns:
                trades_df['Profit/Loss ($)'] = trades_df['Profit/Loss ($)'].apply(
                    lambda x: f"${x:.2f}" if pd.notnull(x) else "")
            
            if 'Profit/Loss (%)' in trades_df.columns:
                trades_df['Profit/Loss (%)'] = trades_df['Profit/Loss (%)'].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
            
            # Add a column to show if entry has stop loss protection
            if args.use_stop_loss:
                # Initialize with object dtype
                trades_df['Stop Loss Status'] = pd.NA
                
                # Only process rows with valid data
                for idx, row in trades_df.iterrows():
                    if row['Action'] == 'BUY':
                        if pd.notnull(row['Stop Loss Price']) and row['Stop Loss Price'] != "":
                            # Extract numeric value from formatted string
                            try:
                                price = float(row['Price'].replace('$', ''))
                                stop_price = float(row['Stop Loss Price'].replace('$', ''))
                                pct_below = ((price - stop_price) / price * 100)
                                trades_df.loc[idx, 'Stop Loss Status'] = f"Protected: {row['Stop Loss Price']} ({pct_below:.2f}% below entry)"
                            except:
                                trades_df.loc[idx, 'Stop Loss Status'] = "Protected"
                        else:
                            trades_df.loc[idx, 'Stop Loss Status'] = "Not Protected"
                    else:
                        trades_df.loc[idx, 'Stop Loss Status'] = "N/A"
            
            # Add a result column for sell trades to indicate if the trade was profitable
            if 'Profit/Loss ($)' in trades_df.columns:
                # Initialize with object dtype
                trades_df['Result'] = pd.NA
                
                # Process only sell trades
                sell_trades = trades_df['Action'] == 'SELL'
                
                # Only process rows with valid profit/loss values
                for idx, row in trades_df[sell_trades].iterrows():
                    if row['Profit/Loss ($)'] != "":
                        try:
                            profit = float(row['Profit/Loss ($)'].replace('$', ''))
                            if profit > 0:
                                trades_df.loc[idx, 'Result'] = "Profit"
                            elif profit < 0:
                                trades_df.loc[idx, 'Result'] = "Loss"
                            else:
                                trades_df.loc[idx, 'Result'] = "Breakeven"
                        except:
                            pass
            
            # Add a column to show if the trade was stopped out but still profitable
            if 'Stop Loss Triggered' in trades_df.columns:
                # Initialize with object dtype before assigning strings
                trades_df['Profitable Stop-Out'] = pd.NA
                
                # Set values only for relevant rows
                sell_and_stopped = (trades_df['Action'] == 'SELL') & (trades_df['Stop Loss Triggered'] == 'Yes')
                
                # Set values based on profitability for stopped out trades
                if 'Result' in trades_df.columns:
                    profitable_stops = sell_and_stopped & (trades_df['Result'] == 'Profit')
                    trades_df.loc[profitable_stops, 'Profitable Stop-Out'] = "Yes (Profitable stop-out)"
                    
                    loss_stops = sell_and_stopped & (trades_df['Result'] == 'Loss')
                    trades_df.loc[loss_stops, 'Profitable Stop-Out'] = "No (Loss on stop-out)"
            
            # Reorder columns for better readability
            if args.use_stop_loss:
                # If using stop loss, prioritize stop loss columns
                column_order = ['Date', 'Action', 'Price', 'Shares', 'Value', 'Commission', 
                                'Trade ID', 'Stop Loss Price', 'Stop Loss Status', 
                                'Stop Loss Triggered', 'Entry Had Stop']
                
                # Add profit/loss columns if they exist
                pl_columns = ['Profit/Loss ($)', 'Profit/Loss (%)', 'Result', 'Profitable Stop-Out']
                for col in pl_columns:
                    if col in trades_df.columns:
                        column_order.append(col)
                
                # Add any remaining columns
                for col in trades_df.columns:
                    if col not in column_order:
                        column_order.append(col)
                
                # Filter to only include columns that actually exist
                column_order = [col for col in column_order if col in trades_df.columns]
                trades_df = trades_df[column_order]
            
            # Save the enhanced dataframe to CSV
            csv_path = f"{backtest_dir}/csv/{ticker}_trades.csv"
            trades_df.to_csv(csv_path, index=False)
            print(f"Trade details saved to {csv_path}")
        else:
            print("No trades were made in this backtest.")
    
    # Generate and save the backtest summary
    summary_path = save_backtest_summary(backtest_dir, args, tickers, strategy_info, results_summary)
    print(f"\nBacktest summary saved to: {summary_path}")
    
    # Print information about the logbook
    if not args.no_logging:
        print("\nStrategy performance has been logged to the logbook system.")
        print("You can analyze past performance using the logbook module.")
        print(f"- Master log: logs/master_log.csv")
        print(f"- Ticker-specific logs: logs/tickers/")
        print(f"- Trade details: logs/trades/")
    
if __name__ == "__main__":
    main() 