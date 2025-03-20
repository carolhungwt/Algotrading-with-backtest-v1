# AlgoTrade with Backtest

A comprehensive algorithmic trading and backtesting system written in Python. This tool allows you to develop, test, and analyze trading strategies on historical stock data.

## Features

- **Multiple Trading Strategies**: Includes popular strategies like Moving Average Crossover, RSI, MACD, and Bollinger Bands
- **Strategy Combination**: Combine multiple strategies with custom weighting or voting methods
- **Separate Buy/Sell Strategies**: Use different strategies for buy signals vs. sell signals
- **Customizable Backtesting**: Test strategies with different parameters, time periods, and trading frequencies
- **Data Retrieval**: Automatically fetch historical stock data from Yahoo Finance
- **Performance Analysis**: Generate comprehensive performance metrics and visualizations
- **Trade Logging**: Track all trades and analyze individual trade performance
- **Visualization**: Generate charts and graphs for backtest results with combined views
- **CSV Export**: Save backtest results for future analysis

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/algotrade-with-backtest.git
cd algotrade-with-backtest
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run a backtest with default parameters:
```
python main.py --tickers AAPL,MSFT,GOOGL --strategies SimpleMovingAverageCrossover
```

Run a backtest with multiple strategies:
```
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy
```

Run a backtest with custom parameters:
```
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover --params '{"SimpleMovingAverageCrossover": {"short_window": 20, "long_window": 50}}'
```

List all available strategies:
```
python main.py --list-strategies
```

Full list of command line options:
```
python main.py --help
```

### Example Usage

#### Basic Backtesting
```bash
# Run a simple backtest on Apple stock using the Moving Average Crossover strategy
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover --period 1y --interval 1d

# Run a backtest on multiple stocks using the RSI strategy with custom parameters
python main.py --tickers AAPL,MSFT,GOOGL --strategies RSIStrategy --params '{"RSIStrategy": {"period": 14, "oversold": 30, "overbought": 70}}' --period 2y

# Combine multiple strategies for a backtest
python main.py --tickers TSLA --strategies SimpleMovingAverageCrossover,RSIStrategy,MACDStrategy --period 1y
```

#### Advanced Signal Combination Methods

```bash
# Use voting method to combine signals from multiple strategies
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy,MACDStrategy --combine-method vote

# Require unanimous agreement among strategies for signals
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy --combine-method unanimous

# Require majority agreement (more than 50%) for signals
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy,MACDStrategy --combine-method majority

# Adjust signal threshold for more sensitive signal generation
python main.py --tickers AAPL --strategies SimpleMovingAverageCrossover,RSIStrategy --signal-threshold 0.1
```

#### Using Separate Buy and Sell Strategies

```bash
# Use different strategies for buy signals vs. sell signals
python main.py --tickers AAPL --separate-signals \
    --buy-strategies SimpleMovingAverageCrossover,MACDStrategy \
    --sell-strategies RSIStrategy,BollingerBandsStrategy \
    --combine-method vote

# Use trend following for entries and overbought/oversold for exits
python main.py --tickers AAPL --separate-signals \
    --buy-strategies SimpleMovingAverageCrossover \
    --sell-strategies RSIStrategy \
    --params '{"RSIStrategy": {"overbought": 75}}'
```

## Signal Combination Methods

The system supports several methods for combining signals from multiple strategies:

1. **Average (default)**: Takes the average of all strategy signals and applies a threshold
2. **Vote**: Each strategy gets one vote, with the majority determining the signal
3. **Unanimous**: Requires all strategies to agree on a signal
4. **Majority**: Requires more than 50% of strategies to agree
5. **Weighted**: Uses weighted averaging based on specified weights

## Separate Buy and Sell Strategies

You can optimize your entries and exits separately by using:
- One set of strategies for generating buy signals
- A different set of strategies for generating sell signals

This is especially useful when certain strategies excel at finding entry points, while others are better at identifying exit points.

## Output

The backtest results will be saved in the `output` directory:
- CSV reports in `output/csv/`
- Charts and visualizations in `output/images/`

The system generates several visualizations for each backtest:
- A combined visualization with portfolio value, trade signals, and drawdown
- Individual detailed charts for each metric
- Trade analysis charts showing profit/loss per trade

## Available Strategies

- **SimpleMovingAverageCrossover**: Generates signals based on the crossover of two moving averages
- **RSIStrategy**: Uses the Relative Strength Index to generate signals based on overbought/oversold conditions
- **MACDStrategy**: Uses the Moving Average Convergence Divergence indicator for signal generation
- **BollingerBandsStrategy**: Generates signals when price crosses Bollinger Bands
- **CompositeStrategy**: Combines multiple strategies with custom weighting

## Creating Custom Strategies

You can create your own strategies by inheriting from the `Strategy` base class and implementing the `generate_signals` method:

```python
from strategy_manager import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, parameters=None):
        default_params = {
            'param1': 10,
            'param2': 20
        }
        parameters = parameters or {}
        self.parameters = {**default_params, **parameters}
        super().__init__(self.parameters)
    
    def generate_signals(self, data):
        # Implement your strategy logic here
        # Return a pandas Series with values:
        # 1 for buy, -1 for sell, 0 for hold
        pass
```

Save your custom strategy in the `strategies` directory to make it available for backtesting.

## License

This project is open-source and available under the MIT License.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used as financial advice or a recommendation to buy or sell any securities. Always do your own research and consult with a licensed financial advisor before making investment decisions. 