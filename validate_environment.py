import numpy as np
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from database.datasets.targets.market_limit_orders_builder import MarketLimitOrdersBuilder
from database.datasets.utils import split_train_test
from environments.environment import TradingEnvironment
from environments.rewards import RewardFunction
from environments.wrappers.tf.utils import validate_environment
from environments.wrappers.tf.tfenv import TFEnvironment

timeframe_len = 12
target_horizon_len = 12
num_eval_samples = 720
fees_percentage = 0.01
episode_steps = 120

df = pd.read_csv(config.dataset_save_filepath.format('BTC'))

targets = MarketLimitOrdersBuilder().build_targets(
    timeframe_len=timeframe_len,
    target_horizon_len=target_horizon_len,
    closes=df['close'].to_numpy(dtype=np.float64),
    highs=df['high'].to_numpy(dtype=np.float64),
    lows=df['low'].to_numpy(dtype=np.float64)
)
data = df[config.regression_features].to_numpy(dtype=np.float32)

scaler = MinMaxScaler()
num_samples_to_scale = data.shape[0] - num_eval_samples - timeframe_len + 1
data[: num_samples_to_scale] = scaler.fit_transform(data[: num_samples_to_scale])
data[num_samples_to_scale:] = scaler.transform(data[num_samples_to_scale:])
inputs = np.float32([data[i: i+timeframe_len] for i in range(data.shape[0]-timeframe_len-target_horizon_len+1)])

x_train, y_train, x_test, y_test = split_train_test(inputs=inputs, targets=targets, num_eval_samples=num_eval_samples)
train_reward_fn = RewardFunction(targets=y_train, fees_percentage=fees_percentage)

env = TradingEnvironment(env_config={
    'states': inputs,
    'reward_fn': train_reward_fn,
    'episode_steps': episode_steps
})
tf_env = TFEnvironment(env=env)
validate_environment(env=tf_env, episodes=10)
