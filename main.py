import argparse
import pandas as pd
from datetime import datetime
import os

from data_handler import DataHandler
from backtest_engine import BacktestEngine
from strategy_manager import StrategyManager
from visualization import Visualizer

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/csv"):
        os.makedirs("output/csv")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")

def main():
    parser = argparse.ArgumentParser(description='AlgoTrade Backtesting System')
    
    # Stock and data parameters
    parser.add_argument('--tickers', type=str, required=True, help='Comma-separated list of stock tickers')
    parser.add_argument('--period', type=str, default='1y', help='Period of historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')
    
    # Strategy parameters
    parser.add_argument('--strategies', type=str, required=True, help='Comma-separated list of strategies to use')
    parser.add_argument('--params', type=str, default='', help='JSON string of strategy parameters')
    
    # Backtest parameters
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital for backtesting')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate per trade')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directories
    create_output_directory()
    
    # Parse tickers and strategies
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    strategy_names = [strategy.strip() for strategy in args.strategies.split(',')]
    
    print(f"Starting backtest for tickers: {', '.join(tickers)}")
    print(f"Using strategies: {', '.join(strategy_names)}")
    
    # Initialize components
    data_handler = DataHandler()
    strategy_manager = StrategyManager()
    visualizer = Visualizer()
    
    # Load strategies
    strategies = strategy_manager.load_strategies(strategy_names, args.params)
    
    # Run backtest for each ticker
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Get data
        data = data_handler.get_stock_data(ticker, args.period, args.interval)
        if data is None or len(data) == 0:
            print(f"No data available for {ticker}, skipping...")
            continue
            
        # Initialize backtest engine
        backtest = BacktestEngine(
            data=data,
            strategies=strategies,
            initial_capital=args.initial_capital,
            commission=args.commission
        )
        
        # Run backtest
        results = backtest.run()
        
        # Display results
        print("\nBacktest Results:")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Number of Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        
        # Visualize results
        visualizer.plot_backtest_results(ticker, data, results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"output/csv/{ticker}_{timestamp}.csv"
        results['trades'].to_csv(csv_path)
        print(f"Trade details saved to {csv_path}")
        
if __name__ == "__main__":
    main() 