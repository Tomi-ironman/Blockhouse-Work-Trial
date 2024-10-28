import pandas as pd
import numpy as np


data = pd.DataFrame({
    'close': np.random.rand(100),  # This should be actual market data 'close' prices
    'volume': np.random.randint(1, 100, 100)  # This should be actual market data 'volume'
})

def twap_strategy(initial_inventory, num_steps):
    return [(initial_inventory // num_steps) for _ in range(num_steps)]

def vwap_strategy(data, initial_inventory):
    total_volume = data['volume'].sum()
    return [(initial_inventory * (volume / total_volume)) for volume in data['volume']]

def backtest_strategy(strategy, data, initial_inventory):
    total_cost = 0
    inventory = initial_inventory
    for step, shares_to_trade in enumerate(strategy):
        execution_price = data.iloc[step]['close']
        total_cost += execution_price * shares_to_trade
        inventory -= shares_to_trade
    return total_cost


def twap_strategy(total_volume, num_periods):
    volume_per_period = total_volume // num_periods
    leftover = total_volume % num_periods
    trades = [volume_per_period] * num_periods
    for i in range(leftover):
        trades[i] += 1
    return trades

def backtest_twap(data, trades):
    total_cost = 0
    for i, trade in enumerate(trades):
        total_cost += trade * data['close'].iloc[i]
    return total_cost


def vwap_strategy(data, total_volume):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['cumulative_volume'] = data['volume'].cumsum()
    data['cumulative_tpv'] = (data['typical_price'] * data['volume']).cumsum()
    data['vwap'] = data['cumulative_tpv'] / data['cumulative_volume']
    vwap_prices = data['vwap']
    total_cost = sum(data['volume'] * data['vwap'])
    return vwap_prices, total_cost