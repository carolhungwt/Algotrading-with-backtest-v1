import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""
    
    def __init__(self, data, strategies=None, buy_strategies=None, sell_strategies=None, 
                 initial_capital=10000.0, commission=0.001, signal_threshold=0.2, 
                 combine_method='average', separate_signals=False,
                 use_stop_loss=False, stop_loss_atr_multiplier=1.0, stop_loss_atr_period=14,
                 scanning_mode=False, scan_for='buy'):
        """
        Initialize the backtest engine.
        
        Args:
            data (pandas.DataFrame): Historical price data with OHLCV columns
            strategies (list, optional): List of Strategy instances to use for both buy and sell signals
            buy_strategies (list, optional): List of Strategy instances to use only for buy signals
            sell_strategies (list, optional): List of Strategy instances to use only for sell signals
            initial_capital (float): Initial capital for the backtest
            commission (float): Commission rate per trade (e.g., 0.001 = 0.1%)
            signal_threshold (float): Threshold for converting continuous signals to discrete (-1, 0, 1)
            combine_method (str): Method to combine multiple strategy signals ('average', 'vote', 'unanimous', 'majority')
            separate_signals (bool): Whether to use separate strategies for buy and sell signals
            use_stop_loss (bool): Whether to use stop loss orders
            stop_loss_atr_multiplier (float): Multiplier for ATR to set stop loss distance
            stop_loss_atr_period (int): Period for ATR calculation
            scanning_mode (bool): Whether to run in scanning mode (no trade execution)
            scan_for (str): The type of signal to scan for in scanning mode ('buy' or 'sell')
        """
        self.data = data
        self.separate_signals = separate_signals
        
        # Handle strategy assignment based on whether using separate signals
        if self.separate_signals:
            # Separate strategies for buy and sell
            self.buy_strategies = buy_strategies or []
            self.sell_strategies = sell_strategies or []
            self.strategies = []  # Not used in this mode
            
            if not self.buy_strategies or not self.sell_strategies:
                raise ValueError("When using separate_signals=True, both buy_strategies and sell_strategies must be provided")
        else:
            # Same strategies for both buy and sell
            self.strategies = strategies or []
            self.buy_strategies = self.strategies  # Reference the same list
            self.sell_strategies = self.strategies  # Reference the same list
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.signal_threshold = signal_threshold
        self.combine_method = combine_method
        
        # Stop loss parameters
        self.use_stop_loss = use_stop_loss
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.stop_loss_atr_period = stop_loss_atr_period
        
        # Scanning mode parameters
        self.scanning_mode = scanning_mode
        self.scan_for = scan_for.lower()
        if self.scan_for not in ['buy', 'sell']:
            raise ValueError("scan_for must be either 'buy' or 'sell'")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Backtest results
        self.portfolio_value = None
        self.positions = None
        self.trades = None
        self.performance_metrics = None
        self.signals_data = None  # For storing signals data in scanning mode
        
        # Validate combine method
        valid_methods = ['average', 'vote', 'unanimous', 'majority', 'weighted']
        if self.combine_method not in valid_methods:
            self.logger.warning(f"Invalid combine_method '{combine_method}'. Using 'average' instead.")
            self.combine_method = 'average'
    
    def run(self):
        """
        Run the backtest.
        
        Returns:
            dict: Dictionary containing backtest results
        """
        # If in scanning mode, use the scanning mode method instead
        if self.scanning_mode:
            return self.run_scanning_mode()
            
        self.logger.info("Starting backtest...")
        
        # Generate trading signals
        signals = self._generate_signals()
        
        # Calculate ATR if stop loss is enabled
        atr = None
        if self.use_stop_loss:
            atr = self._calculate_atr()
        
        # Initialize variables for tracking portfolio
        self.portfolio_value = pd.Series(self.initial_capital, index=self.data.index)
        self.trades = pd.DataFrame(columns=['date', 'action', 'price', 'shares', 'value', 'commission', 'trade_id', 'stop_loss_price', 'stopped_out'])
        self.position = 0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.trade_id = 0
        self.stopped_out_count = 0  # Track number of trades stopped out
        
        # Variables for stop loss tracking
        current_position_entry_price = 0
        current_stop_loss_price = 0
        
        # Iterate through the data and execute trades
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            current_price = self.data['Close'][i]
            
            # Check if we hit a stop loss
            hit_stop_loss = False
            if self.use_stop_loss and self.position > 0 and not np.isnan(current_stop_loss_price):
                # Check if the low price of the day went below the stop loss
                if self.data['Low'][i] <= current_stop_loss_price:
                    # Execute stop loss - use stop price or today's opening price, whichever is more conservative
                    execute_price = max(current_stop_loss_price, self.data['Open'][i])
                    self._execute_trade(date, 'SELL', execute_price, self.position, stopped_out=True)
                    self.position = 0
                    hit_stop_loss = True
                    self.stopped_out_count += 1  # Increment stopped out counter
                    self.logger.info(f"Stop loss triggered at {execute_price:.2f} on {date}")
            
            # If we didn't hit stop loss, check for regular signals
            if not hit_stop_loss:
                signal = signals[i]
                
                # Execute buy signal - only when we have no position
                if signal > self.signal_threshold and self.position == 0:
                    # Calculate shares to buy using all available cash, accounting for commission
                    # Formula: shares = cash / (price * (1 + commission))
                    shares_to_buy = int(self.cash / (current_price * (1 + self.commission)))
                    
                    if shares_to_buy > 0:
                        # Calculate stop loss if applicable
                        stop_loss_price = None
                        if self.use_stop_loss and i >= self.stop_loss_atr_period and not np.isnan(atr[i]):
                            current_atr = atr[i]
                            stop_distance = current_atr * self.stop_loss_atr_multiplier
                            stop_loss_price = current_price - stop_distance
                            current_stop_loss_price = stop_loss_price
                            self.logger.info(f"Setting stop loss at {stop_loss_price:.2f} (ATR: {current_atr:.2f})")
                        
                        # Execute trade with stop loss price
                        self._execute_trade(date, 'BUY', current_price, shares_to_buy, stop_loss_price=stop_loss_price)
                        self.position = shares_to_buy
                        current_position_entry_price = current_price
                
                # Execute sell signal - only when we have a position
                elif signal < -self.signal_threshold and self.position > 0:
                    # Sell all shares we currently hold
                    self._execute_trade(date, 'SELL', current_price, self.position)
                    self.position = 0
                    current_stop_loss_price = 0  # Reset stop loss
            
            # Update portfolio value
            self.equity = self.cash + (self.position * current_price)
            self.portfolio_value[i] = self.equity
        
        # Liquidate any remaining position at the end of the backtest
        if self.position > 0:
            last_date = self.data.index[-1]
            last_price = self.data['Close'].iloc[-1]
            self._execute_trade(last_date, 'SELL', last_price, self.position)
            self.position = 0
        
        # Calculate backtest metrics
        results = self._calculate_metrics()
        
        # Add stop loss metrics to results
        if self.use_stop_loss:
            results['stopped_out_count'] = self.stopped_out_count
        
        self.logger.info("Backtest completed.")
        return results
    
    def _generate_signals(self):
        """
        Generate trading signals based on the strategies.
        
        Returns:
            pandas.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        if self.separate_signals:
            # Generate buy and sell signals separately
            buy_signals = self._generate_combined_signals(self.buy_strategies, signal_type='buy')
            sell_signals = self._generate_combined_signals(self.sell_strategies, signal_type='sell')
            
            # Combine buy and sell signals into a single series
            combined_signal = pd.Series(0, index=self.data.index)
            combined_signal[buy_signals == 1] = 1  # Buy signals
            combined_signal[sell_signals == -1] = -1  # Sell signals
            
            # In case of conflict (both buy and sell on the same day), prioritize sell
            # This can be customized based on preference
            conflict_days = (buy_signals == 1) & (sell_signals == -1)
            if conflict_days.any():
                self.logger.warning(f"Conflicts detected on {conflict_days.sum()} days. Prioritizing sell signals.")
                combined_signal[conflict_days] = -1
        else:
            # Use the same strategies for both buy and sell signals
            combined_signal = self._generate_combined_signals(self.strategies)
        
        return combined_signal
    
    def _generate_combined_signals(self, strategies, signal_type=None):
        """
        Generate and combine trading signals from provided strategies.
        
        Args:
            strategies (list): List of Strategy instances to use
            signal_type (str, optional): Type of signal to generate ('buy', 'sell', or None for both)
            
        Returns:
            pandas.Series: Combined trading signals
        """
        strategy_count = len(strategies)
        signal_desc = f"{signal_type} " if signal_type else ""
        self.logger.info(f"Generating {signal_desc}signals using {strategy_count} strategies with method: {self.combine_method}")
        
        # If there are no strategies, return a series of zeros (no signals)
        if not strategies:
            return pd.Series(0, index=self.data.index)
        
        # If only one strategy, use its signals
        if strategy_count == 1:
            signals = strategies[0].generate_signals(self.data)
            
            # If generating only buy or sell signals, filter out the other type
            if signal_type == 'buy':
                signals = signals.where(signals > 0, 0)  # Keep only buy signals
            elif signal_type == 'sell':
                signals = signals.where(signals < 0, 0)  # Keep only sell signals
                
            return signals
        
        # Generate signals from each strategy
        all_signals = []
        for strategy in strategies:
            signals = strategy.generate_signals(self.data)
            
            # If generating only buy or sell signals, filter out the other type
            if signal_type == 'buy':
                signals = signals.where(signals > 0, 0)  # Keep only buy signals
            elif signal_type == 'sell':
                signals = signals.where(signals < 0, 0)  # Keep only sell signals
                
            all_signals.append(signals)
        
        # Convert to DataFrame for easier manipulation
        signals_df = pd.concat(all_signals, axis=1)
        
        # Combine signals based on the specified method
        if self.combine_method == 'average':
            # Take the average of all signals
            combined_signals = signals_df.mean(axis=1)
            
            # Convert to discrete signals (-1, 0, 1) using threshold
            discrete_signals = pd.Series(0, index=combined_signals.index)
            
            if signal_type != 'sell':  # Include buy signals
                discrete_signals[combined_signals > self.signal_threshold] = 1
            
            if signal_type != 'buy':  # Include sell signals
                discrete_signals[combined_signals < -self.signal_threshold] = -1
            
        elif self.combine_method == 'vote':
            # Each strategy gets one vote
            # Count the number of buy, sell, and hold signals
            buys = (signals_df > 0).sum(axis=1)
            sells = (signals_df < 0).sum(axis=1)
            
            # Use the most common signal
            discrete_signals = pd.Series(0, index=signals_df.index)
            
            if signal_type != 'sell':  # Include buy signals
                discrete_signals[buys > sells] = 1
            
            if signal_type != 'buy':  # Include sell signals
                discrete_signals[sells > buys] = -1
            
        elif self.combine_method == 'unanimous':
            # All strategies must agree
            if signal_type != 'sell':  # Check for unanimous buy signals
                buy_unanimous = (signals_df > 0).all(axis=1)
            else:
                buy_unanimous = pd.Series(False, index=signals_df.index)
                
            if signal_type != 'buy':  # Check for unanimous sell signals
                sell_unanimous = (signals_df < 0).all(axis=1)
            else:
                sell_unanimous = pd.Series(False, index=signals_df.index)
            
            discrete_signals = pd.Series(0, index=signals_df.index)
            discrete_signals[buy_unanimous] = 1
            discrete_signals[sell_unanimous] = -1
            
        elif self.combine_method == 'majority':
            # Majority of strategies must agree (more than 50%)
            discrete_signals = pd.Series(0, index=signals_df.index)
            
            if signal_type != 'sell':  # Check for majority buy signals
                buys = (signals_df > 0).sum(axis=1)
                discrete_signals[buys > strategy_count / 2] = 1
                
            if signal_type != 'buy':  # Check for majority sell signals
                sells = (signals_df < 0).sum(axis=1)
                discrete_signals[sells > strategy_count / 2] = -1
            
        elif self.combine_method == 'weighted':
            # If strategies have weights, use them
            weighted_signals = None
            
            # Check if using a CompositeStrategy
            if strategy_count == 1 and hasattr(strategies[0], 'weighting') and hasattr(strategies[0], 'strategies'):
                # Get signals directly from the composite strategy
                weighted_signals = strategies[0].generate_signals(self.data)
                
                # Apply signal type filtering if needed
                if signal_type == 'buy':
                    weighted_signals = weighted_signals.where(weighted_signals > 0, 0)
                elif signal_type == 'sell':
                    weighted_signals = weighted_signals.where(weighted_signals < 0, 0)
            else:
                # Manual weighting - default to equal weights
                weights = [1.0 / strategy_count] * strategy_count
                
                # Generate weighted signals
                weighted_signals = pd.Series(0, index=self.data.index)
                for i, strategy in enumerate(strategies):
                    signals = strategy.generate_signals(self.data)
                    
                    # Filter by signal type if needed
                    if signal_type == 'buy':
                        signals = signals.where(signals > 0, 0)
                    elif signal_type == 'sell':
                        signals = signals.where(signals < 0, 0)
                        
                    weighted_signals += signals * weights[i]
            
            # Convert to discrete signals based on thresholds
            discrete_signals = pd.Series(0, index=weighted_signals.index)
            
            if signal_type != 'sell':  # Include buy signals
                discrete_signals[weighted_signals > self.signal_threshold] = 1
            
            if signal_type != 'buy':  # Include sell signals
                discrete_signals[weighted_signals < -self.signal_threshold] = -1
        
        return discrete_signals
    
    def _execute_trade(self, date, action, price, shares, stop_loss_price=None, stopped_out=False):
        """
        Execute a trade and record it.
        
        Args:
            date (datetime): Date of the trade
            action (str): 'BUY' or 'SELL'
            price (float): Price at which to execute the trade
            shares (int): Number of shares to trade
            stop_loss_price (float, optional): Stop loss price for this trade
            stopped_out (bool, optional): Whether this trade was stopped out
        """
        value = price * shares
        commission_amount = value * self.commission
        
        if action == 'BUY':
            self.cash -= (value + commission_amount)
            self.trade_id += 1
        elif action == 'SELL':
            self.cash += (value - commission_amount)
            
        # Record the trade
        trade_info = {
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'value': value,
            'commission': commission_amount,
            'trade_id': self.trade_id,
            'stop_loss_price': stop_loss_price,
            'stopped_out': stopped_out
        }
        
        self.trades = pd.concat([self.trades, pd.DataFrame([trade_info])], ignore_index=True)
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics from backtest results.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        portfolio_value = self.portfolio_value
        trades = self.trades
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_value.iloc[-1]
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Annualized return
        days = (self.data.index[-1] - self.data.index[0]).days
        if days > 0:
            annual_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
        else:
            annual_return = 0
        
        # Daily returns
        daily_returns = portfolio_value.pct_change().dropna()
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len(trades[trades['action'] == 'SELL'])  # Count sell transactions
        
        # Win rate
        if not trades.empty and 'profit_loss' in trades.columns:
            winning_trades = trades[(trades['action'] == 'SELL') & (trades['profit_loss'] > 0)]
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Stop loss statistics
        stopped_out_count = self.stopped_out_count if hasattr(self, 'stopped_out_count') else 0
        
        # Calculate P&L for each trade
        if total_trades > 0:
            # Match buy and sell trades by trade_id
            buy_trades = trades[trades['action'] == 'BUY'].set_index('trade_id')
            sell_trades = trades[trades['action'] == 'SELL'].set_index('trade_id')
            
            # For each trade_id, calculate P&L
            for trade_id in buy_trades.index:
                if trade_id in sell_trades.index:
                    buy_price = buy_trades.loc[trade_id, 'price']
                    buy_commission = buy_trades.loc[trade_id, 'commission']
                    buy_value = buy_trades.loc[trade_id, 'value']
                    
                    sell_price = sell_trades.loc[trade_id, 'price']
                    sell_commission = sell_trades.loc[trade_id, 'commission']
                    sell_value = sell_trades.loc[trade_id, 'value']
                    
                    # Calculate profit/loss in dollars
                    profit_loss = sell_value - buy_value - buy_commission - sell_commission
                    
                    # Calculate profit/loss percentage relative to buy value
                    profit_loss_pct = (profit_loss / buy_value) * 100
                    
                    # Record in the trades DataFrame
                    sell_idx = trades[(trades['trade_id'] == trade_id) & (trades['action'] == 'SELL')].index
                    if len(sell_idx) > 0:
                        trades.loc[sell_idx, 'profit_loss'] = profit_loss
                        trades.loc[sell_idx, 'profit_loss_pct'] = profit_loss_pct
        
        # Create the results dictionary
        results = {
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "portfolio_value": portfolio_value,
            "trades": trades
        }
        
        # Add stop loss metrics if enabled
        if self.use_stop_loss:
            results['stopped_out_count'] = stopped_out_count
            results['stop_loss_atr_multiplier'] = self.stop_loss_atr_multiplier
            results['stop_loss_atr_period'] = self.stop_loss_atr_period
            
            # Calculate P&L for normal vs stopped out trades if we have P&L data
            if not trades.empty and 'profit_loss' in trades.columns:
                # Calculate P&L for stopped out trades
                stopped_trades = trades[(trades['action'] == 'SELL') & (trades['stopped_out'] == True)]
                if not stopped_trades.empty:
                    results['stopped_out_pl'] = stopped_trades['profit_loss'].sum()
                    results['avg_stopped_pl'] = stopped_trades['profit_loss'].mean()
                    results['stopped_win_rate'] = (stopped_trades['profit_loss'] > 0).mean() * 100
                else:
                    results['stopped_out_pl'] = 0
                    results['avg_stopped_pl'] = 0
                    results['stopped_win_rate'] = 0
                    
                # Calculate P&L for normal exit trades
                normal_trades = trades[(trades['action'] == 'SELL') & (trades['stopped_out'] == False)]
                if not normal_trades.empty:
                    results['normal_pl'] = normal_trades['profit_loss'].sum()
                    results['avg_normal_pl'] = normal_trades['profit_loss'].mean()
                    results['normal_win_rate'] = (normal_trades['profit_loss'] > 0).mean() * 100
                else:
                    results['normal_pl'] = 0
                    results['avg_normal_pl'] = 0
                    results['normal_win_rate'] = 0
        
        return results
    
    def plot_results(self):
        """Plot backtest results (for quick visualization during development)."""
        if self.portfolio_value is None:
            print("No backtest results to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_value.index, self.portfolio_value.values)
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # Plot trades on price chart
        plt.subplot(2, 1, 2)
        plt.plot(self.data.index, self.data['Close'])
        
        # Plot buy signals
        buys = self.trades[self.trades['action'] == 'BUY']
        plt.scatter(buys['date'], buys['price'], marker='^', color='g', label='Buy')
        
        # Plot sell signals
        sells = self.trades[self.trades['action'] == 'SELL']
        plt.scatter(sells['date'], sells['price'], marker='v', color='r', label='Sell')
        
        plt.title('Price & Trades')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def _calculate_atr(self):
        """
        Calculate Average True Range (ATR) for stop loss calculation.
        
        Returns:
            pandas.Series: ATR values
        """
        self.logger.info("Calculating ATR...")
        
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        # Calculate True Range
        tr = pd.Series(0, index=self.data.index)
        tr[1:] = pd.Series(high[1:] - low[1:], index=self.data.index[1:])
        tr = tr.fillna(0)
        
        # Calculate ATR
        atr = pd.Series(0, index=self.data.index)
        # First ATR value is just the average of first n periods
        atr.iloc[self.stop_loss_atr_period - 1] = tr[:self.stop_loss_atr_period].mean()

        # Calculate ATR using smoothing method
        for i in range(self.stop_loss_atr_period, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (self.stop_loss_atr_period - 1) + tr.iloc[i]) / self.stop_loss_atr_period
        
        return atr

    def run_scanning_mode(self):
        """
        Run in scanning mode to identify buy or sell signals without executing trades.
        This mode focuses on signal generation without portfolio tracking.
        
        Returns:
            dict: Dictionary containing scanning results
        """
        self.logger.info(f"Starting backtest scanning mode for {self.scan_for.upper()} signals...")
        
        # Generate trading signals
        raw_signals = self._generate_signals()
        
        # Initialize signal tracking data structures
        signal_dates = []
        signal_prices = []
        signal_strengths = []  # To track the strength of each signal
        
        # Create a DataFrame to store all market data along with signals
        self.signals_data = self.data.copy()
        self.signals_data['signal'] = raw_signals
        
        # Iterate through the data and identify signals
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            current_price = self.data['Close'][i]
            signal = raw_signals[i]
            
            # Track buy signals
            if self.scan_for == 'buy' and signal > self.signal_threshold:
                signal_dates.append(date)
                signal_prices.append(current_price)
                signal_strengths.append(signal)
                self.logger.info(f"Buy signal detected at {date}: price={current_price:.2f}, strength={signal:.4f}")
                
            # Track sell signals
            elif self.scan_for == 'sell' and signal < -self.signal_threshold:
                signal_dates.append(date)
                signal_prices.append(current_price)
                signal_strengths.append(abs(signal))  # Store absolute value for easier comparison
                self.logger.info(f"Sell signal detected at {date}: price={current_price:.2f}, strength={abs(signal):.4f}")
        
        # Create signal dataframe
        signals_df = pd.DataFrame({
            'date': signal_dates,
            'price': signal_prices,
            'strength': signal_strengths,
            'signal_type': self.scan_for
        })
        
        # Calculate scanning metrics
        total_signals = len(signals_df)
        signal_frequency = total_signals / len(self.data) if len(self.data) > 0 else 0
        avg_signal_strength = signals_df['strength'].mean() if total_signals > 0 else 0
        
        # Calculate price changes after signals to evaluate accuracy
        if total_signals > 0:
            # Add columns for future price changes (N days later)
            for days in [1, 5, 10, 20]:
                signals_df[f'price_change_{days}d'] = 0.0
                signals_df[f'price_change_{days}d_pct'] = 0.0
                
                for idx, row in signals_df.iterrows():
                    signal_date = row['date']
                    signal_idx = self.data.index.get_loc(signal_date)
                    
                    # Check if we have enough future data
                    if signal_idx + days < len(self.data):
                        future_price = self.data['Close'].iloc[signal_idx + days]
                        price_change = future_price - row['price']
                        price_change_pct = (future_price / row['price'] - 1) * 100
                        
                        signals_df.loc[idx, f'price_change_{days}d'] = price_change
                        signals_df.loc[idx, f'price_change_{days}d_pct'] = price_change_pct
            
            # Calculate accuracy metrics based on price movement direction
            # For buy signals, accuracy means price went up
            # For sell signals, accuracy means price went down
            for days in [1, 5, 10, 20]:
                col = f'price_change_{days}d_pct'
                if self.scan_for == 'buy':
                    accuracy = (signals_df[col] > 0).mean() * 100
                else:  # sell
                    accuracy = (signals_df[col] < 0).mean() * 100
                signals_df[f'accuracy_{days}d'] = accuracy
                
        # Create the results dictionary
        results = {
            "signals": signals_df,
            "signals_data": self.signals_data,
            "total_signals": total_signals,
            "signal_frequency": signal_frequency,
            "avg_signal_strength": avg_signal_strength,
            "scan_for": self.scan_for
        }
        
        self.logger.info(f"Scanning completed. Found {total_signals} {self.scan_for} signals.")
        
        # If there are signals, log additional information
        if total_signals > 0:
            self.logger.info(f"Average signal strength: {avg_signal_strength:.4f}")
            for days in [1, 5, 10, 20]:
                if f'accuracy_{days}d' in signals_df.columns:
                    accuracy = signals_df[f'accuracy_{days}d'].iloc[0]
                    self.logger.info(f"{days}-day accuracy: {accuracy:.2f}%")
        
        return results

    def plot_scanning_results(self, ticker=None):
        """
        Plot scanning mode results, showing price chart with signals
        and metrics for signal evaluation.
        
        Args:
            ticker (str, optional): Ticker symbol for display purposes
        """
        if self.signals_data is None:
            print("No scanning results to plot.")
            return
            
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data with signals on first subplot
        axs[0].plot(self.data.index, self.data['Close'], label='Close Price', color='blue', alpha=0.7)
        
        # Find indices where signals were generated
        if self.scan_for == 'buy':
            signal_indices = self.signals_data[self.signals_data['signal'] > self.signal_threshold].index
            marker_color = 'green'
            signal_label = 'Buy Signal'
        else:  # sell signals
            signal_indices = self.signals_data[self.signals_data['signal'] < -self.signal_threshold].index
            marker_color = 'red'
            signal_label = 'Sell Signal'
        
        # Plot signals as markers
        if len(signal_indices) > 0:
            axs[0].scatter(
                signal_indices,
                self.signals_data.loc[signal_indices, 'Close'],
                color=marker_color,
                marker='^' if self.scan_for == 'buy' else 'v',
                s=100,
                label=signal_label
            )
            
            # Add signal strength annotations for a few signals
            if len(signal_indices) <= 10:
                # If few signals, annotate all
                signals_to_annotate = signal_indices
            else:
                # If many signals, annotate a sample
                signals_to_annotate = signal_indices[::max(1, len(signal_indices) // 10)]
            
            for idx in signals_to_annotate:
                signal_val = self.signals_data.loc[idx, 'signal']
                axs[0].annotate(
                    f"{abs(signal_val):.2f}",
                    (idx, self.signals_data.loc[idx, 'Close']),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                
        # Add additional price data context
        if 'SMA_20' not in self.data.columns and len(self.data) > 20:
            self.signals_data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            axs[0].plot(self.data.index, self.signals_data['SMA_20'], label='20-day SMA', 
                      color='purple', linestyle='--', alpha=0.6)
        elif 'SMA_20' in self.data.columns:
            axs[0].plot(self.data.index, self.data['SMA_20'], label='20-day SMA', 
                      color='purple', linestyle='--', alpha=0.6)
        
        # Label the top plot
        title = f"{self.scan_for.capitalize()} Signal Scanning"
        if ticker:
            title += f" for {ticker}"
        axs[0].set_title(title)
        axs[0].set_ylabel('Price')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Second subplot: Signal strength distribution or price change after signals
        if hasattr(self, 'signals') and len(self.signals) > 0:
            # Plot distribution of future price changes after signals
            for days in [5, 20]:
                col = f'price_change_{days}d_pct'
                if col in self.signals.columns:
                    axs[1].hist(
                        self.signals[col], 
                        bins=20, 
                        alpha=0.6, 
                        label=f"{days}-day Change"
                    )
            
            axs[1].set_title('Price Change Distribution After Signals')
            axs[1].set_xlabel('Price Change (%)')
            axs[1].set_ylabel('Frequency')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()
        else:
            # If we don't have detailed signal data, show signal strength over time
            signal_column = 'signal'
            if signal_column in self.signals_data.columns:
                # For buy signals, positive values matter
                if self.scan_for == 'buy':
                    signal_values = self.signals_data[signal_column].apply(lambda x: max(0, x))
                # For sell signals, negative values matter (convert to positive for display)
                else:
                    signal_values = self.signals_data[signal_column].apply(lambda x: max(0, -x))
                
                axs[1].plot(self.data.index, signal_values, color='orange', label='Signal Strength')
                axs[1].axhline(y=self.signal_threshold, color='red', linestyle='--', 
                             label=f'Signal Threshold ({self.signal_threshold})')
                
                axs[1].set_title(f"{self.scan_for.capitalize()} Signal Strength Over Time")
                axs[1].set_ylabel('Signal Strength')
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()
                
        # Adjust layout and show plot
        plt.tight_layout()
        return fig 