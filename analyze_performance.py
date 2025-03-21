#!/usr/bin/env python3
"""
Analysis tool for the logbook system - helps analyze strategy performance
across different backtest runs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from logbook import Logbook

def analyze_ticker(ticker, logbook, metrics=None):
    """Analyze performance for a specific ticker"""
    ticker_data = logbook.get_ticker_performance(ticker)
    
    if ticker_data.empty:
        print(f"No data found for ticker {ticker}")
        return
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE ANALYSIS FOR {ticker}")
    print(f"{'='*80}")
    print(f"Total backtest runs: {len(ticker_data)}")
    
    # Display summary statistics
    if metrics:
        for metric in metrics:
            if metric in ticker_data.columns:
                print(f"\n{metric.replace('_', ' ').title()} Statistics:")
                print(f"  Mean: {ticker_data[metric].mean():.2f}")
                print(f"  Median: {ticker_data[metric].median():.2f}")
                print(f"  Min: {ticker_data[metric].min():.2f}")
                print(f"  Max: {ticker_data[metric].max():.2f}")
    
    # Show best strategies
    best_return = ticker_data.sort_values('total_return', ascending=False).head(5)
    print("\nTop 5 Strategies by Total Return:")
    for i, (_, row) in enumerate(best_return.iterrows(), 1):
        print(f"  {i}. Return: {row['total_return']:.2f}% - Buy: {row['buy_strategy']}, Sell: {row['sell_strategy']}")
        print(f"     Stop Loss: {'Enabled' if row['use_stop_loss'] else 'Disabled'}, " 
              f"Trades: {row['num_trades']}, Win Rate: {row['win_rate']:.2f}%")
    
    # Show best win rates
    best_win_rate = ticker_data.sort_values('win_rate', ascending=False).head(5)
    print("\nTop 5 Strategies by Win Rate:")
    for i, (_, row) in enumerate(best_win_rate.iterrows(), 1):
        print(f"  {i}. Win Rate: {row['win_rate']:.2f}% - Buy: {row['buy_strategy']}, Sell: {row['sell_strategy']}")
        print(f"     Return: {row['total_return']:.2f}%, Trades: {row['num_trades']}")
    
    return ticker_data

def analyze_strategy(strategy, logbook):
    """Analyze performance for a specific strategy"""
    strategy_data = logbook.get_strategy_performance(strategy)
    
    if strategy_data.empty:
        print(f"No data found for strategy {strategy}")
        return
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE ANALYSIS FOR STRATEGY: {strategy}")
    print(f"{'='*80}")
    print(f"Total backtest runs: {len(strategy_data)}")
    
    # Group by ticker
    ticker_groups = strategy_data.groupby('ticker')
    
    print("\nPerformance by Ticker:")
    for ticker, group in ticker_groups:
        avg_return = group['total_return'].mean()
        avg_win_rate = group['win_rate'].mean()
        total_runs = len(group)
        
        print(f"  {ticker}: {total_runs} runs, Avg Return: {avg_return:.2f}%, Avg Win Rate: {avg_win_rate:.2f}%")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Average Return: {strategy_data['total_return'].mean():.2f}%")
    print(f"  Average Win Rate: {strategy_data['win_rate'].mean():.2f}%")
    print(f"  Average Trades per Run: {strategy_data['num_trades'].mean():.1f}")
    
    return strategy_data

def plot_ticker_strategy_comparison(ticker, logbook, output_dir='analysis_plots'):
    """Create plots comparing strategy performance for a ticker"""
    ticker_data = logbook.get_ticker_performance(ticker)
    
    if ticker_data.empty:
        print(f"No data found for ticker {ticker}")
        return
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Identify unique strategies
    buy_strategies = []
    for strategies in ticker_data['buy_strategy'].unique():
        if pd.notna(strategies):
            buy_strategies.extend([s.strip() for s in strategies.split(',')])
    buy_strategies = list(set(buy_strategies))
    
    # Plot return by strategy
    plt.figure(figsize=(12, 8))
    
    # Group data by strategy
    strategy_returns = []
    strategy_names = []
    
    for strategy in buy_strategies:
        strategy_data = ticker_data[ticker_data['buy_strategy'].str.contains(strategy, na=False)]
        if not strategy_data.empty:
            strategy_returns.append(strategy_data['total_return'].values)
            strategy_names.append(strategy)
    
    if strategy_returns:
        plt.boxplot(strategy_returns, labels=strategy_names)
        plt.title(f'Strategy Return Comparison for {ticker}')
        plt.ylabel('Total Return (%)')
        plt.xlabel('Strategy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ticker}_strategy_comparison.png")
        print(f"Saved strategy comparison plot to {output_dir}/{ticker}_strategy_comparison.png")
    
    # Plot win rate vs return
    plt.figure(figsize=(10, 6))
    plt.scatter(ticker_data['win_rate'], ticker_data['total_return'], alpha=0.7)
    
    # Add labels for top performers
    top_performers = ticker_data.nlargest(5, 'total_return')
    for _, row in top_performers.iterrows():
        plt.annotate(f"{row['buy_strategy']}", 
                     (row['win_rate'], row['total_return']),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'Win Rate vs. Return for {ticker}')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}_winrate_vs_return.png")
    print(f"Saved win rate vs return plot to {output_dir}/{ticker}_winrate_vs_return.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze backtest performance from logbook')
    
    parser.add_argument('--ticker', type=str, help='Analyze a specific ticker')
    parser.add_argument('--strategy', type=str, help='Analyze a specific strategy')
    parser.add_argument('--list-tickers', action='store_true', help='List all tickers in the logbook')
    parser.add_argument('--list-strategies', action='store_true', help='List all strategies in the logbook')
    parser.add_argument('--best', action='store_true', help='Show best performing strategies')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Initialize logbook
    logbook = Logbook()
    
    # Load master log for overall analysis
    try:
        master_log = pd.read_csv(logbook.master_log_path)
    except Exception:
        print("Error: Could not load master log. Make sure you've run some backtests first.")
        return
    
    if args.list_tickers:
        tickers = master_log['ticker'].unique()
        print("\nTickers with backtest data:")
        for ticker in sorted(tickers):
            ticker_count = len(master_log[master_log['ticker'] == ticker])
            print(f"  {ticker}: {ticker_count} backtest runs")
    
    if args.list_strategies:
        # Collect all strategies from buy and sell columns
        all_strategies = set()
        
        for col in ['buy_strategy', 'sell_strategy']:
            for strategies in master_log[col].dropna():
                all_strategies.update([s.strip() for s in strategies.split(',')])
        
        print("\nStrategies used in backtests:")
        for strategy in sorted(all_strategies):
            strategy_count = len(master_log[(master_log['buy_strategy'].str.contains(strategy, na=False)) | 
                                           (master_log['sell_strategy'].str.contains(strategy, na=False))])
            print(f"  {strategy}: {strategy_count} backtest runs")
    
    if args.ticker:
        ticker_data = analyze_ticker(args.ticker, logbook, 
                                    metrics=['total_return', 'win_rate', 'num_trades'])
        
        if args.plot and ticker_data is not None:
            plot_ticker_strategy_comparison(args.ticker, logbook)
    
    if args.strategy:
        analyze_strategy(args.strategy, logbook)
    
    if args.best:
        # Show best overall performers
        print("\nBest Overall Performers:")
        best_overall = master_log.sort_values('total_return', ascending=False).head(10)
        
        for i, (_, row) in enumerate(best_overall.iterrows(), 1):
            print(f"  {i}. {row['ticker']} - Return: {row['total_return']:.2f}%")
            print(f"     Buy: {row['buy_strategy']}, Sell: {row['sell_strategy']}")
            print(f"     Stop Loss: {'Enabled' if row['use_stop_loss'] else 'Disabled'}, " 
                  f"Trades: {row['num_trades']}, Win Rate: {row['win_rate']:.2f}%")
    
    if not any([args.ticker, args.strategy, args.list_tickers, args.list_strategies, args.best]):
        print("No analysis options specified. Use --help to see available options.")

if __name__ == "__main__":
    main() 