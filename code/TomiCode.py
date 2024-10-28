import pandas as pd
import numpy as np
# from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging

@dataclass
class MarketState:
    """Represents the current state of the market"""
    volatility: float
    spread: float
    volume: float
    price: float
    time_of_day: float

class AdaptiveTradingStrategy:
    def __init__(self, data: pd.DataFrame, risk_aversion: float = 1.0):
        """
        Initialize the adaptive trading strategy with sophisticated cost models.
        
        Args:
            data (pd.DataFrame): Market data including bid/ask prices and volumes
            risk_aversion (float): Risk aversion parameter for the utility function
        """
        self.data = data
        self.risk_aversion = risk_aversion
        self.logger = logging.getLogger(__name__)
        
        # Initialize market impact parameters based on Almgren-Chriss model
        self.temporary_impact = 1.4e-6  # Temporary price impact
        self.permanent_impact = 2.5e-6  # Permanent price impact
        self.sigma = self._calculate_volatility()
        
    def _calculate_volatility(self, window: int = 20) -> float:
        """
        Calculate rolling volatility using log returns
        """
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        return log_returns.rolling(window=window).std().mean()

    def _calculate_market_state(self, idx: int) -> MarketState:
        """
        Calculate current market state including volatility, spread, and volume metrics
        """
        current_data = self.data.iloc[idx]
        
        # Calculate spread
        spread = current_data['ask_price_1'] - current_data['bid_price_1']
        
        # Calculate normalized volume
        volume = current_data['volume'] / self.data['volume'].rolling(30).mean().iloc[idx]
        
        # Calculate time of day factor (U-shaped pattern)
        minutes_from_open = idx % 390  # Assuming 390-minute trading day
        time_factor = 1.0 + 0.1 * (np.exp(-minutes_from_open/60) + np.exp(-(390-minutes_from_open)/60))
        
        return MarketState(
            volatility=self._calculate_volatility(),
            spread=spread,
            volume=volume,
            price=current_data['close'],
            time_of_day=time_factor
        )

    def _calculate_optimal_trade_size(self, 
                                    remaining_shares: float,
                                    remaining_time: int,
                                    market_state: MarketState) -> float:
        """
        Calculate optimal trade size using Almgren-Chriss framework with adjustments
        for market conditions
        """
        if remaining_time <= 0:
            return remaining_shares
            
        # Adjust impact parameters based on market state
        adjusted_temp_impact = self.temporary_impact * (
            1 + 0.5 * (market_state.volatility / self.sigma - 1)
        ) * market_state.time_of_day
        
        # Calculate optimal trade size using AC model with modifications
        trade_size = (remaining_shares / remaining_time) * (
            1 + self.risk_aversion * self.sigma * np.sqrt(remaining_time/390)
        )
        
        # Adjust for volume profile
        trade_size *= np.sqrt(market_state.volume)
        
        # Ensure trade size doesn't exceed remaining shares
        return min(trade_size, remaining_shares)

    def generate_adaptive_schedule(self, 
                                 initial_inventory: int,
                                 preferred_timeframe: int = 390) -> pd.DataFrame:
        """
        Generate an adaptive trading schedule based on market conditions
        
        Args:
            initial_inventory (int): Initial number of shares to trade
            preferred_timeframe (int): Trading horizon in minutes
            
        Returns:
            pd.DataFrame: Trading schedule with timestamps and share sizes
        """
        remaining_inventory = initial_inventory
        trades = []
        
        for step in range(min(len(self.data), preferred_timeframe)):
            market_state = self._calculate_market_state(step)
            
            # Calculate optimal trade size for current conditions
            optimal_shares = self._calculate_optimal_trade_size(
                remaining_inventory,
                preferred_timeframe - step,
                market_state
            )
            
            # Round to nearest whole share
            trade_size = int(np.round(optimal_shares))
            
            if trade_size > 0:
                remaining_inventory -= trade_size
                trade = {
                    'timestamp': self.data.iloc[step]['timestamp'],
                    'step': step,
                    'price': self.data.iloc[step]['close'],
                    'shares': trade_size,
                    'inventory': remaining_inventory,
                    'market_volatility': market_state.volatility,
                    'spread': market_state.spread,
                    'volume_factor': market_state.volume
                }
                trades.append(trade)
                
            if remaining_inventory <= 0:
                break
                
        return pd.DataFrame(trades)

    def calculate_implementation_shortfall(self,
                                        trades: pd.DataFrame,
                                        initial_price: float) -> Dict[str, float]:
        """
        Calculate implementation shortfall and other execution metrics
        """
        total_cost = 0
        permanent_impact = 0
        temporary_impact = 0
        
        for idx, trade in trades.iterrows():
            shares = trade['shares']
            exec_price = trade['price']
            
            # Calculate temporary impact cost
            temp_impact = self.temporary_impact * shares * exec_price
            temporary_impact += temp_impact
            
            # Calculate permanent impact
            perm_impact = self.permanent_impact * shares * exec_price
            permanent_impact += perm_impact
            
            # Total transaction cost
            trade_cost = (exec_price - initial_price) * shares + temp_impact + perm_impact
            total_cost += trade_cost
            
        return {
            'total_cost': total_cost,
            'implementation_shortfall': total_cost / (initial_price * trades['shares'].sum()),
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact
        }

class EnhancedBenchmark(Benchmark):
    """
    Enhanced version of the original Benchmark class with additional features
    """
    def __init__(self, data):
        super().__init__(data)
        self.adaptive_strategy = AdaptiveTradingStrategy(data)
        
    def get_adaptive_trades(self, initial_inventory: int, preferred_timeframe: int = 390) -> pd.DataFrame:
        """
        Generate trades using the adaptive strategy
        """
        return self.adaptive_strategy.generate_adaptive_schedule(
            initial_inventory,
            preferred_timeframe
        )
        
    def compare_strategies(self, initial_inventory: int, preferred_timeframe: int = 390) -> Dict:
        """
        Compare performance of different trading strategies
        """
        # Get trades for each strategy
        twap_trades = self.get_twap_trades(self.data, initial_inventory, preferred_timeframe)
        vwap_trades = self.get_vwap_trades(self.data, initial_inventory, preferred_timeframe)
        adaptive_trades = self.get_adaptive_trades(initial_inventory, preferred_timeframe)
        
        initial_price = self.data['close'].iloc[0]
        
        # Calculate costs for each strategy
        results = {}
        for name, trades in [
            ('TWAP', twap_trades),
            ('VWAP', vwap_trades),
            ('Adaptive', adaptive_trades)
        ]:
            slippage, market_impact = self.simulate_strategy(trades, self.data, preferred_timeframe)
            implementation_shortfall = self.adaptive_strategy.calculate_implementation_shortfall(
                trades, initial_price
            )
            
            results[name] = {
                'total_slippage': sum(slippage),
                'total_market_impact': sum(market_impact),
                'implementation_shortfall': implementation_shortfall['implementation_shortfall'],
                'trade_count': len(trades),
                'average_trade_size': trades['shares'].mean()
            }
            
        return results