#!/usr/bin/env python3
"""
Test Features Script for AlgoTrade

This script demonstrates the various features of the AlgoTrade system,
focusing on the debug mode and scanning mode capabilities.
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n{'='*80}")
        print(f"  {description}")
        print(f"{'='*80}")
    
    print(f"\nRunning: {' '.join(command)}\n")
    start_time = time.time()
    
    # Run the command and stream the output
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output to console
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    end_time = time.time()
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds with exit code {process.returncode}")
    return process.returncode

def test_debug_mode(args):
    """Run tests demonstrating the debug mode features."""
    
    # Make sure debug directories exist
    os.makedirs("debug/csv", exist_ok=True)
    os.makedirs("debug/images", exist_ok=True)
    
    # Basic debug mode with single strategy
    run_command(
        ["python", "main.py", "--tickers", args.tickers, 
         "--strategies", "TripleMASlope", 
         "--period", "3mo", "--interval", "1d",
         "--debug"],
        "Test 1: Basic debug mode with TripleMASlope strategy"
    )
    
    # Debug mode with separate buy/sell strategies
    if args.run_all:
        run_command(
            ["python", "main.py", "--tickers", args.tickers,
             "--separate-signals",
             "--buy-strategies", "TripleMASlope", 
             "--sell-strategies", "TripleMASlope",
             "--period", "3mo", "--interval", "1d",
             "--debug"],
            "Test 2: Debug mode with separate buy/sell strategies"
        )
    
    # Debug with custom parameters
    if args.run_all:
        params = '{"fast_window": 5, "mid_window": 8, "slow_window": 15, "slope_threshold": 0.03}'
        run_command(
            ["python", "main.py", "--tickers", args.tickers,
             "--strategies", "TripleMASlope",
             "--params", params,
             "--period", "3mo", "--interval", "1d",
             "--debug"],
            "Test 3: Debug mode with custom parameters"
        )

    print("\nDebug mode tests completed. Check the debug/csv and debug/images directories for output.")

def test_scanning_mode(args):
    """Run tests demonstrating the scanning mode features."""
    
    # Make sure scanning output directories exist
    os.makedirs("backtest_scanning_output/buy_signals/csv", exist_ok=True)
    os.makedirs("backtest_scanning_output/buy_signals/images", exist_ok=True)
    os.makedirs("backtest_scanning_output/sell_signals/csv", exist_ok=True)
    os.makedirs("backtest_scanning_output/sell_signals/images", exist_ok=True)
    
    # Basic scanning mode for buy signals
    run_command(
        ["python", "scan_signals.py", "scan",
         "--tickers", args.tickers,
         "--strategies", "TripleMASlope",
         "--period", "1y",
         "--scan-for", "buy"],
        "Test 1: Scanning for buy signals with TripleMASlope"
    )
    
    # Scanning mode for sell signals
    if args.run_all:
        run_command(
            ["python", "scan_signals.py", "scan",
             "--tickers", args.tickers,
             "--strategies", "TripleMASlope",
             "--period", "1y",
             "--scan-for", "sell"],
            "Test 2: Scanning for sell signals with TripleMASlope"
        )
    
    # Analyze scan results
    run_command(
        ["python", "scan_signals.py", "analyze"],
        "Analyze scan results"
    )
    
    print("\nScanning mode tests completed. Check the backtest_scanning_output directory for results.")

def test_all(args):
    """Run both debug and scanning mode tests."""
    test_debug_mode(args)
    test_scanning_mode(args)

def main():
    parser = argparse.ArgumentParser(description="Test AlgoTrade Features")
    parser.add_argument("--mode", choices=["debug", "scan", "all"], default="all",
                      help="Which mode to test (debug, scan, or all)")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOG",
                      help="Comma-separated list of tickers to test with")
    parser.add_argument("--run-all", action="store_true",
                      help="Run all tests (otherwise runs only basic tests)")
    
    args = parser.parse_args()
    
    print(f"\nStarting feature tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing with tickers: {args.tickers}")
    
    if args.mode == "debug":
        test_debug_mode(args)
    elif args.mode == "scan":
        test_scanning_mode(args)
    else:
        test_all(args)
    
    print(f"\nAll tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 