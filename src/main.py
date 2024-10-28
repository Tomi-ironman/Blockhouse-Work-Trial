import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# Import custom functions and classes
from trading_strategies import twap_strategy, vwap_strategy, backtest_strategy
from custom_env import CustomTradingEnv

# Example data creation for demonstration
data = pd.DataFrame({
    'close': np.random.rand(100), 
    'volume': np.random.randint(1, 100, 100) 
})

# Create and check the custom trading environment
env = CustomTradingEnv(data)
check_env(env)  

# Define the model
model = SAC('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)  

# Optional: Save the trained model
model.save("sac_trading_model")

# Backtest the SAC model
obs = env.reset()
done = False
sac_total_cost = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    sac_total_cost += rewards

# Output the results
print(f"SAC Model Cost: {sac_total_cost}")

# Backtest TWAP
twap_trades = twap_strategy(1000, len(data))
twap_cost = backtest_strategy(twap_trades, data, 1000)

# Backtest VWAP
vwap_trades = vwap_strategy(data, 1000)
vwap_cost = backtest_strategy(vwap_trades, data, 1000)

print(f"TWAP Cost: {twap_cost}")
print(f"VWAP Cost: {vwap_cost}")