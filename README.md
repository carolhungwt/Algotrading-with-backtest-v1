# AlgoTrade with Backtest

A Python-based algorithmic trading platform with backtest capabilities to test and evaluate trading strategies using historical data.

## Features

- **Customizable Trading Strategies**: Implement and test various trading strategies including Moving Average Crossover, RSI, MACD, Bollinger Bands, and more
- **Strategy Combination**: Combine multiple strategies using voting, averaging, or unanimous consent methods
- **Separate Buy/Sell Strategies**: Use different strategies for buy and sell signals to optimize entry and exit points
- **ATR-Based Stop Loss**: Automatically set dynamic stop loss levels based on market volatility
- **Comprehensive Backtesting**: Test your strategies against historical data with detailed performance metrics
- **Visualization Tools**: Generate charts and graphs to analyze trading performance
- **Trade Analysis**: Analyze trade details, profit/loss per trade, and trade duration impact on returns
- **Organized Results**: Save all backtest results in dedicated folders for easy comparison and reference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/algotrade-with-backtest.git
cd algotrade-with-backtest
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run backtests using the command line interface:

```bash
python main.py --tickers AAPL,MSFT --strategies SimpleMovingAverageCrossover --period 1y
```

### Basic Parameters:

- `--tickers`: Comma-separated list of stock tickers to backtest
- `--period`: Period of historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `--interval`: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
- `--initial-capital`: Initial capital for backtesting (default: 10000)
- `--commission`: Commission rate per trade (default: 0.001 or 0.1%)

### Strategy Selection:

#### Using the Same Strategy for Buy and Sell Signals:
```bash
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover
```

#### Using Multiple Strategies Together:
```bash
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy
```

#### Using Separate Strategies for Buy and Sell Signals:
```bash
python main.py --tickers AAPL --separate-signals --buy-strategies SimpleMovingAverageCrossover --sell-strategies RSIStrategy
```

### Signal Combination Methods:

When using multiple strategies, you can specify how to combine their signals:

```bash
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy --combine-method vote
```

Available combination methods:
- `average`: Average the signals from all strategies (default)
- `vote`: Each strategy gets one vote, majority wins
- `unanimous`: All strategies must agree for a signal
- `majority`: More than half of strategies must agree
- `weighted`: Weight strategies by importance (when using a CompositeStrategy)

### Stop Loss Feature:

Protect your trades with ATR-based stop loss levels that adapt to market volatility:

```bash
# Enable stop loss with default settings (1.0 × ATR)
python main.py --tickers AAPL --strategies MovingAverageCrossover --use-stop-loss

# Customize ATR multiplier (example: 2.0 × ATR for wider stops)
python main.py --tickers AAPL --strategies MovingAverageCrossover --use-stop-loss --stop-loss-atr-multiplier 2.0

# Customize ATR period (default is 14 periods)
python main.py --tickers AAPL --strategies MovingAverageCrossover --use-stop-loss --stop-loss-atr-period 21
```

The stop loss is placed at `entry_price - (ATR × multiplier)` for long positions. When the price drops below this level, the position is automatically sold.

## Available Strategies

- `SimpleMovingAverageCrossover`: Buy when short MA crosses above long MA, sell when it crosses below
- `RSIStrategy`: Buy when RSI crosses above oversold threshold, sell when it crosses below overbought threshold
- `MACDStrategy`: Buy/sell based on MACD line crossing signal line
- `BollingerBandsStrategy`: Buy when price crosses below lower band, sell when it crosses above upper band
- `CompositeStrategy`: Combine multiple strategies with custom weightings
- `TripleMASlope`: Buy/sell based on triple moving average crossovers with slope confirmation to identify trending markets

To list all available strategies:
```bash
python main.py --list-strategies
```

## Customizing Strategy Parameters

You can customize strategy parameters using JSON format:

```bash
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover --params '{"short_window": 20, "long_window": 50}'
```

## Debug Mode

The platform includes a debugging system to help analyze strategy performance in detail:

```bash
# Enable debug mode for any strategy
python main.py --tickers AAPL --strategies TripleMASlope --debug

# Specify a custom directory for debug files
python main.py --tickers AAPL --strategies TripleMASlope --debug --debug-dir my_debug_folder
```

### Debug Features

- **Detailed Logging**: Captures signal generation details, slope values, and market conditions
- **Visualization**: Generates charts showing moving averages, slopes, and signal points
- **Trade Analysis**: Saves CSV files with detailed information about each trade
- **Analysis Tools**: Use the included analysis script to evaluate debug data:

```bash
# Analyze the latest debug run with plots
python analyze_triple_ma.py --latest --plots

# Analyze a specific debug file
python analyze_triple_ma.py --file triple_ma_debug_20230615_143022.csv
```

## Example Use Cases

### 1. Simple Moving Average Crossover

```bash
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover --period 2y
```

### 2. Combining RSI and Bollinger Bands with Voting

```bash
python main.py --tickers MSFT --strategies RSIStrategy,BollingerBandsStrategy --combine-method vote
```

### 3. Using Different Strategies for Buy and Sell with Stop Loss

```bash
python main.py --tickers TSLA --separate-signals --buy-strategies BollingerBandsStrategy --sell-strategies RSIStrategy --use-stop-loss --stop-loss-atr-multiplier 1.5
```

### 4. Multiple Tickers Comparison

```bash
python main.py --tickers AAPL,MSFT,GOOG --strategies MACDStrategy
```

### 5. Triple Moving Average with Slope Confirmation

```bash
# Run with default parameters (6, 10, 20 periods)
python main.py --tickers AAPL --strategies TripleMASlope

# With custom parameters
python main.py --tickers AAPL --strategies TripleMASlope --params '{"TripleMASlope": {"fast_window": 5, "mid_window": 8, "slow_window": 15, "slope_threshold": 0.05, "slope_period": 3}}'

# With debug mode enabled
python main.py --tickers AAPL --strategies TripleMASlope --debug
```

The Triple MA Slope strategy uses three moving averages with slope confirmation to identify trending markets:

- **Buy Signal**: Mid MA (10) crosses above Slow MA (20) with positive slope, and market is not in an equivocal state
- **Sell Signal**: Fast MA (6) crosses below Mid MA (10) with negative slope, and market is not in an equivocal state

Parameters:
- `fast_window`: Period for fast moving average (default: 6)
- `mid_window`: Period for middle moving average (default: 10)
- `slow_window`: Period for slow moving average (default: 20)
- `slope_period`: Periods used to calculate slope (default: 3)
- `slope_threshold`: Threshold for determining significant slope (default: 0.01)
- `equivocal_threshold`: Threshold for determining sideways markets (default: 0.002)

## Backtest Results

Each backtest run creates a timestamped directory in the `output` folder containing:

- Summary text file with all test parameters and results
- Trade CSV files with detailed trade information
- Performance visualizations including:
  - Portfolio value chart
  - Trading signals with stop loss levels
  - Drawdown analysis
  - Trade profit/loss analysis

## Adding Custom Strategies

1. Create a new strategy file in the `strategies` directory
2. Inherit from the `Strategy` base class
3. Implement the `generate_signals` method

Example:
```python
from strategies.strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, params=None):
        self.params = params or {}
        # Initialize your strategy parameters
        
    def generate_signals(self, data):
        # Your strategy logic here
        # Return a pandas Series with buy (1), sell (-1), and hold (0) signals
        return signals
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data access
- [pandas](https://pandas.pydata.org/) for data manipulation
- [matplotlib](https://matplotlib.org/) for visualization
- [numpy](https://numpy.org/) for numerical operations 