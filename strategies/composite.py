import pandas as pd
import numpy as np
from strategy_manager import Strategy

class CompositeStrategy(Strategy):
    """
    A strategy that combines multiple trading strategies with optional weighting.
    
    Generates signals based on the weighted average of signals from the combined strategies.
    """
    
    def __init__(self, strategies, weighting=None):
        """
        Initialize the composite strategy.
        
        Args:
            strategies (list): List of strategy instances to combine
            weighting (list, optional): List of weights for each strategy. 
                                       If None, all strategies are weighted equally.
        """
        super().__init__()
        
        self.strategies = strategies
        
        # If weighting is not provided, use equal weights
        if weighting is None:
            self.weighting = [1.0 / len(strategies)] * len(strategies)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weighting)
            self.weighting = [w / total_weight for w in weighting]
        
        self.name = "CompositeStrategy"
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the combined strategies.
        
        Args:
            data (pandas.DataFrame): Historical price data
            
        Returns:
            pandas.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        if not self.strategies:
            return pd.Series(0, index=data.index)
        
        # Generate signals for each strategy
        all_signals = []
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(data)
            # Apply weight
            weighted_signals = signals * self.weighting[i]
            all_signals.append(weighted_signals)
        
        # Combine signals
        combined_signals = pd.concat(all_signals, axis=1).sum(axis=1)
        
        # Convert to discrete signals based on thresholds
        discrete_signals = pd.Series(0, index=combined_signals.index)
        discrete_signals[combined_signals > 0.2] = 1
        discrete_signals[combined_signals < -0.2] = -1
        
        return discrete_signals
    
    def add_strategy(self, strategy, weight=None):
        """
        Add a new strategy to the composite strategy.
        
        Args:
            strategy (Strategy): Strategy instance to add
            weight (float, optional): Weight for the new strategy. 
                                    If None, weights are recalculated to be equal.
        """
        self.strategies.append(strategy)
        
        if weight is None:
            # Reset to equal weights
            self.weighting = [1.0 / len(self.strategies)] * len(self.strategies)
        else:
            # Normalize existing weights to make room for the new weight
            current_sum = sum(self.weighting)
            self.weighting = [w * (1 - weight) / current_sum for w in self.weighting]
            self.weighting.append(weight)
    
    def remove_strategy(self, index):
        """
        Remove a strategy from the composite strategy.
        
        Args:
            index (int): Index of the strategy to remove
        """
        if 0 <= index < len(self.strategies):
            del self.strategies[index]
            del self.weighting[index]
            
            if self.strategies:
                # Recalculate weights to sum to 1
                total_weight = sum(self.weighting)
                self.weighting = [w / total_weight for w in self.weighting]
    
    def __str__(self):
        strategy_str = ", ".join([f"{s.__str__()}" for s in self.strategies])
        weight_str = ", ".join([f"{w:.2f}" for w in self.weighting])
        return f"Composite Strategy with:\n  - Strategies: {strategy_str}\n  - Weights: {weight_str}" 