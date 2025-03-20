import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""
    
    def __init__(self, data, strategies=None, buy_strategies=None, sell_strategies=None, 
                 initial_capital=10000.0, commission=0.001, signal_threshold=0.2, 
                 combine_method='average', separate_signals=False):
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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Backtest results
        self.portfolio_value = None
        self.positions = None
        self.trades = None
        self.performance_metrics = None
        
        # Validate combine method
        valid_methods = ['average', 'vote', 'unanimous', 'majority', 'weighted']
        if self.combine_method not in valid_methods:
            self.logger.warning(f"Invalid combine_method '{combine_method}'. Using 'average' instead.")
            self.combine_method = 'average'
    
    def run(self):
        """
        Run the backtest using the provided strategies.
        
        Returns:
            dict: Dictionary containing backtest results
        """
        self.logger.info("Starting backtest...")
        
        if self.separate_signals:
            self.logger.info("Using separate strategies for buy and sell signals")
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
        
        # Execute the signals on the historical data
        results = self._execute_signals(combined_signal)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)
        
        self.logger.info(f"Backtest complete. Final portfolio value: ${metrics['final_value']:.2f}")
        
        return {
            "positions": results["positions"],
            "portfolio_value": results["portfolio_value"],
            "trades": results["trades"],
            "signals": combined_signal,
            "final_value": metrics["final_value"],
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"]
        }
    
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
    
    def _execute_signals(self, signals):
        """
        Execute trading signals on the historical data.
        
        Args:
            signals (pandas.Series): Trading signals (1=buy, -1=sell, 0=hold)
            
        Returns:
            dict: Dictionary containing backtest results
        """
        self.logger.info("Executing signals...")
        
        # Initialize portfolio tracking variables
        portfolio_value = pd.Series(index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        cash = pd.Series(self.initial_capital, index=self.data.index)
        trades = []
        
        # Process each day in the backtest
        for i, date in enumerate(self.data.index):
            # Skip the first day (no previous data)
            if i == 0:
                portfolio_value[date] = self.initial_capital
                continue
            
            yesterday = self.data.index[i-1]
            current_price = self.data['Close'][date]
            current_position = positions[yesterday]
            current_cash = cash[yesterday]
            current_signal = signals[date]
            
            # Determine action based on signal and current position
            # Buy signal and not already long
            if current_signal == 1 and current_position == 0:
                # Calculate max shares to buy with available cash
                max_shares = int(current_cash / (current_price * (1 + self.commission)))
                
                # Record the trade if shares are bought
                if max_shares > 0:
                    positions[date] = max_shares
                    trade_cost = max_shares * current_price * (1 + self.commission)
                    cash[date] = current_cash - trade_cost
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': max_shares,
                        'value': max_shares * current_price,
                        'commission': max_shares * current_price * self.commission,
                        'total': trade_cost
                    })
                else:
                    # Not enough cash to buy
                    positions[date] = current_position
                    cash[date] = current_cash
            
            # Sell signal and currently long
            elif current_signal == -1 and current_position > 0:
                positions[date] = 0  # Sell all shares
                
                # Calculate proceeds from sale
                trade_value = current_position * current_price
                commission_cost = trade_value * self.commission
                trade_proceeds = trade_value - commission_cost
                cash[date] = current_cash + trade_proceeds
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': current_position,
                    'value': trade_value,
                    'commission': commission_cost,
                    'total': trade_proceeds
                })
            
            # No action or invalid action
            else:
                positions[date] = current_position
                cash[date] = current_cash
            
            # Calculate portfolio value
            portfolio_value[date] = positions[date] * current_price + cash[date]
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade P&L
        if not trades_df.empty:
            # Label each trade with a unique ID
            trades_df['trade_id'] = None
            current_trade_id = 0
            
            for i in range(len(trades_df)):
                if trades_df.iloc[i]['action'] == 'BUY':
                    current_trade_id += 1
                    trades_df.iloc[i, trades_df.columns.get_loc('trade_id')] = current_trade_id
                
                elif trades_df.iloc[i]['action'] == 'SELL' and i > 0:
                    # Find the most recent BUY
                    for j in range(i-1, -1, -1):
                        if trades_df.iloc[j]['action'] == 'BUY' and pd.isna(trades_df.iloc[j]['trade_id']):
                            trades_df.iloc[i, trades_df.columns.get_loc('trade_id')] = current_trade_id
                            break
            
            # Calculate profit/loss for each complete trade
            buy_trades = trades_df[trades_df['action'] == 'BUY'].set_index('trade_id')
            sell_trades = trades_df[trades_df['action'] == 'SELL'].set_index('trade_id')
            
            for trade_id in buy_trades.index:
                if trade_id in sell_trades.index:
                    buy_value = buy_trades.loc[trade_id, 'total']
                    sell_value = sell_trades.loc[trade_id, 'total']
                    profit = sell_value - buy_value
                    profit_pct = (profit / buy_value) * 100
                    
                    # Add P&L to the sell trade
                    idx = trades_df[
                        (trades_df['trade_id'] == trade_id) & 
                        (trades_df['action'] == 'SELL')
                    ].index
                    
                    trades_df.loc[idx, 'profit_loss'] = profit
                    trades_df.loc[idx, 'profit_loss_pct'] = profit_pct
        
        return {
            "positions": positions,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "trades": trades_df
        }
    
    def _calculate_performance_metrics(self, results):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results (dict): Dictionary containing backtest results
            
        Returns:
            dict: Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        portfolio_value = results["portfolio_value"]
        trades = results["trades"]
        
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
        total_trades = len(trades) // 2  # Each trade has a buy and sell
        
        # Win rate
        if not trades.empty and 'profit_loss' in trades.columns:
            winning_trades = trades[trades['profit_loss'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        else:
            win_rate = 0
        
        return {
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades
        } 