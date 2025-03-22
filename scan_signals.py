#!/usr/bin/env python3
"""
Signal Scanning Tool for AlgoTrade

This script provides a simplified interface for running the backtest system
in scanning mode to identify buy or sell signals without executing trades.
It's designed to help evaluate strategies for identifying market bottoms (buy signals)
and market tops (sell signals).
"""

import argparse
import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

def run_scan(args):
    """Run signal scanning using the main AlgoTrade system."""
    
    # Base command
    cmd = [
        "python", "main.py",
        "--tickers", args.tickers,
        "--period", args.period,
        "--interval", args.interval,
        "--scanning-mode",
        "--scan-for", args.scan_for,
        "--signal-threshold", str(args.signal_threshold),
        "--combine-method", args.combine_method
    ]
    
    # Add strategies based on scan type
    if args.separate_signals:
        cmd.extend(["--separate-signals"])
        
        if args.scan_for == 'buy':
            cmd.extend(["--buy-strategies", args.strategies])
            
            # Add sell strategies if specified
            if args.sell_strategies:
                cmd.extend(["--sell-strategies", args.sell_strategies])
        else:  # scan_for == 'sell'
            cmd.extend(["--sell-strategies", args.strategies])
            
            # Add buy strategies if specified
            if args.buy_strategies:
                cmd.extend(["--buy-strategies", args.buy_strategies])
    else:
        cmd.extend(["--strategies", args.strategies])
    
    # Add custom parameters if provided
    if args.params:
        cmd.extend(["--params", args.params])
    
    # Add scan output directory if specified
    if args.output_dir:
        cmd.extend(["--scan-output-dir", args.output_dir])
        
    print(f"Running scan command: {' '.join(cmd)}")
    subprocess.run(cmd)

def analyze_scans(args):
    """Analyze existing scan results from the scanning output directory."""
    
    # Determine which directories to scan
    if args.scan_for != 'both':
        # Only scan one signal type
        signal_dirs = [os.path.join(args.output_dir, f"{args.scan_for}_signals")]
    else:
        # Scan both buy and sell signal directories
        signal_dirs = [
            os.path.join(args.output_dir, "buy_signals"),
            os.path.join(args.output_dir, "sell_signals")
        ]
    
    # Process each signal directory
    results = []
    for signal_dir in signal_dirs:
        if not os.path.exists(signal_dir):
            print(f"Warning: Signal directory {signal_dir} does not exist. Skipping.")
            continue
            
        signal_type = os.path.basename(signal_dir).split('_')[0]  # 'buy' or 'sell'
        
        # Find all CSV files with signal data
        csv_dir = os.path.join(signal_dir, "csv")
        signal_files = glob.glob(os.path.join(csv_dir, f"*_{signal_type}_signals.csv"))
        
        if not signal_files:
            print(f"No signal data found in {csv_dir}. Skipping.")
            continue
            
        print(f"Found {len(signal_files)} {signal_type} signal files for analysis.")
        
        # Process each file
        for signal_file in signal_files:
            ticker = os.path.basename(signal_file).split('_')[0]
            
            try:
                # Read signal data
                df = pd.read_csv(signal_file)
                
                # Extract key metrics
                signal_count = len(df)
                avg_strength = df['strength'].mean() if 'strength' in df.columns else 0
                
                # Find accuracy metrics (named like 'accuracy_5d')
                accuracy_cols = [col for col in df.columns if col.startswith('accuracy_')]
                accuracies = {}
                
                for col in accuracy_cols:
                    # Extract the day number from column name
                    days = int(col.split('_')[1][:-1])  # Convert "accuracy_5d" to 5
                    accuracies[days] = df[col].iloc[0] if not df.empty else 0
                
                # Extract future price changes where available
                price_changes = {}
                for col in df.columns:
                    if col.startswith('price_change_') and col.endswith('_pct'):
                        days = int(col.split('_')[2][:-4])  # Convert "price_change_5d_pct" to 5
                        changes = df[col].dropna().tolist()
                        if changes:
                            price_changes[days] = changes
                
                # Build result record
                result = {
                    'ticker': ticker,
                    'signal_type': signal_type,
                    'signal_count': signal_count,
                    'avg_strength': avg_strength,
                    'accuracies': accuracies,
                    'price_changes': price_changes,
                    'file_path': signal_file
                }
                
                results.append(result)
                print(f"Processed {ticker} {signal_type} signals: {signal_count} signals found.")
                
            except Exception as e:
                print(f"Error processing {signal_file}: {str(e)}")
    
    if not results:
        print("No scan results found for analysis.")
        return
    
    # Generate comparison report
    generate_comparison_report(results, args.output_dir)

def generate_comparison_report(results, output_dir):
    """Generate a comparison report of scan results."""
    
    # Create an output directory for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"scan_analysis_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Group results by signal type
    buy_results = [r for r in results if r['signal_type'] == 'buy']
    sell_results = [r for r in results if r['signal_type'] == 'sell']
    
    # Create summary tables
    summary_rows = []
    
    for result in results:
        ticker = result['ticker']
        signal_type = result['signal_type']
        signal_count = result['signal_count']
        avg_strength = result['avg_strength']
        
        # Get accuracies for common time periods
        accuracies = {}
        for days in [1, 5, 10, 20]:
            accuracies[days] = result['accuracies'].get(days, float('nan'))
        
        # Add to summary rows
        summary_rows.append({
            'Ticker': ticker,
            'Signal Type': signal_type,
            'Signal Count': signal_count,
            'Avg. Strength': avg_strength,
            '1-day Accuracy (%)': accuracies.get(1, float('nan')),
            '5-day Accuracy (%)': accuracies.get(5, float('nan')),
            '10-day Accuracy (%)': accuracies.get(10, float('nan')),
            '20-day Accuracy (%)': accuracies.get(20, float('nan'))
        })
    
    if summary_rows:
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary to CSV
        summary_path = os.path.join(report_dir, "signal_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to {summary_path}")
        
        # Generate comparison charts
        
        # 1. Signal count comparison
        if len(summary_df) > 1:  # Only if we have multiple results
            plt.figure(figsize=(12, 6))
            
            # Use different colors for buy and sell
            colors = ['green' if t == 'buy' else 'red' for t in summary_df['Signal Type']]
            
            # Create bar chart of signal counts
            sns.barplot(x='Ticker', y='Signal Count', hue='Signal Type', data=summary_df, palette='Set1')
            plt.title('Signal Count Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure
            count_path = os.path.join(report_dir, "signal_count_comparison.png")
            plt.savefig(count_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Accuracy comparison
            plt.figure(figsize=(14, 8))
            
            # Reshape data for accuracy comparison
            accuracy_cols = [col for col in summary_df.columns if 'Accuracy' in col]
            if accuracy_cols:
                # Melt the dataframe to get accuracy by time period
                melted = pd.melt(summary_df, 
                                id_vars=['Ticker', 'Signal Type'],
                                value_vars=accuracy_cols,
                                var_name='Time Period', 
                                value_name='Accuracy (%)')
                
                # Plot grouped bar chart
                sns.catplot(x='Ticker', y='Accuracy (%)', hue='Time Period', 
                          col='Signal Type', data=melted, kind='bar',
                          height=6, aspect=1.5, legend_out=False)
                
                plt.suptitle('Signal Accuracy by Time Period', y=1.02, fontsize=16)
                plt.tight_layout()
                
                # Save figure
                accuracy_path = os.path.join(report_dir, "signal_accuracy_comparison.png")
                plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            # 3. Price change distribution after signals
            # For each result with price changes, create a histogram
            for result in results:
                if result['price_changes']:
                    ticker = result['ticker']
                    signal_type = result['signal_type']
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Plot histograms for different time periods
                    for days, changes in result['price_changes'].items():
                        if changes:
                            plt.hist(changes, bins=20, alpha=0.6, label=f"{days}-day Change")
                    
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    plt.title(f"Price Change Distribution After {signal_type.capitalize()} Signals - {ticker}")
                    plt.xlabel('Price Change (%)')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save figure
                    dist_path = os.path.join(report_dir, f"{ticker}_{signal_type}_price_distribution.png")
                    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Analysis report generated in {report_dir}")

def main():
    parser = argparse.ArgumentParser(description='AlgoTrade Signal Scanning Tool')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Run signal scanning')
    scan_parser.add_argument('--tickers', type=str, required=True, help='Comma-separated list of stock tickers')
    scan_parser.add_argument('--strategies', type=str, required=True, help='Comma-separated list of strategies to use')
    scan_parser.add_argument('--scan-for', type=str, choices=['buy', 'sell'], default='buy', help='Type of signal to scan for')
    scan_parser.add_argument('--period', type=str, default='1y', help='Period of historical data')
    scan_parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    scan_parser.add_argument('--signal-threshold', type=float, default=0.2, help='Signal threshold')
    scan_parser.add_argument('--combine-method', type=str, default='average', choices=['average', 'vote', 'unanimous', 'majority', 'weighted'], help='Signal combination method')
    scan_parser.add_argument('--separate-signals', action='store_true', help='Use separate strategies for buy and sell signals')
    scan_parser.add_argument('--buy-strategies', type=str, help='Strategies for buy signals (when using separate signals)')
    scan_parser.add_argument('--sell-strategies', type=str, help='Strategies for sell signals (when using separate signals)')
    scan_parser.add_argument('--params', type=str, help='JSON string of strategy parameters')
    scan_parser.add_argument('--output-dir', type=str, help='Output directory for scan results')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing scan results')
    analyze_parser.add_argument('--scan-for', type=str, choices=['buy', 'sell', 'both'], default='both', help='Type of signals to analyze')
    analyze_parser.add_argument('--output-dir', type=str, default='backtest_scanning_output', help='Base directory containing scan results')
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        run_scan(args)
    elif args.command == 'analyze':
        analyze_scans(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 