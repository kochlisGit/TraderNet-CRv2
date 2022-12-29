import numpy as np
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    def __init__(
            self,
            timeframe_size: int,
            target_horizon_len: int,
            highs: np.ndarray,
            lows: np.ndarray,
            closes: np.ndarray,
            fees_percentage: float,
            verbose: bool = False
    ):
        rewards_fn = self._build_reward_fn(
            timeframe_size=timeframe_size,
            target_horizon_len=target_horizon_len,
            highs=highs,
            lows=lows,
            closes=closes
        )

        fees = np.log((1 - fees_percentage)/(1 + fees_percentage))
        rewards_fn[:, 0:2] += fees
        hold_rewards = -rewards_fn.max(axis=1)
        hold_rewards[hold_rewards > 0] = 0
        self._rewards_fn = np.hstack((rewards_fn, np.expand_dims(hold_rewards, axis=-1)))

        if verbose:
            print(f'Rewards: {self._rewards_fn.shape}')

    @property
    def reward_fn(self) -> np.ndarray:
        return self._rewards_fn

    @reward_fn.setter
    def reward_fn(self, reward_fn: np.ndarray):
        self._rewards_fn = reward_fn

    def __call__(self, i: int, action: int) -> float:
        return self.get_reward(i=i, action=action)

    def get_reward(self, i: int, action: int) -> float:
        return self._rewards_fn[i, action]

    def get_reward_fn_shape(self):
        return self._rewards_fn.shape

    @abstractmethod
    def _build_reward_fn(
            self,
            timeframe_size: int,
            target_horizon_len: int,
            highs: np.ndarray,
            lows: np.ndarray,
            closes: np.ndarray
    ) -> np.ndarray:
        pass
