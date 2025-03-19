import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""
    
    def __init__(self, data, strategies, initial_capital=10000.0, commission=0.001):
        """
        Initialize the backtest engine.
        
        Args:
            data (pandas.DataFrame): Historical price data with OHLCV columns
            strategies (list): List of Strategy instances to test
            initial_capital (float): Initial capital for the backtest
            commission (float): Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.data = data
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Backtest results
        self.portfolio_value = None
        self.positions = None
        self.trades = None
        self.performance_metrics = None
    
    def run(self):
        """
        Run the backtest using the provided strategies.
        
        Returns:
            dict: Dictionary containing backtest results
        """
        self.logger.info("Starting backtest...")
        
        # Generate signals for each strategy
        combined_signal = self._generate_combined_signals()
        
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
    
    def _generate_combined_signals(self):
        """
        Generate and combine trading signals from all strategies.
        
        Returns:
            pandas.Series: Combined trading signals
        """
        self.logger.info(f"Generating signals using {len(self.strategies)} strategies...")
        
        # If there are no strategies, return a series of zeros (no signals)
        if not self.strategies:
            return pd.Series(0, index=self.data.index)
        
        # If only one strategy, use its signals
        if len(self.strategies) == 1:
            return self.strategies[0].generate_signals(self.data)
        
        # For multiple strategies, take the average of signals
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(self.data)
            all_signals.append(signals)
        
        # Convert to DataFrame for easier averaging
        signals_df = pd.concat(all_signals, axis=1)
        
        # Average the signals across strategies
        combined_signals = signals_df.mean(axis=1)
        
        # Convert to discrete signals (-1, 0, 1)
        # Threshold values can be adjusted as needed
        discrete_signals = pd.Series(0, index=combined_signals.index)
        discrete_signals[combined_signals > 0.2] = 1
        discrete_signals[combined_signals < -0.2] = -1
        
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