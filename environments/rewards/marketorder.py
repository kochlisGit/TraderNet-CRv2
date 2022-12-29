import numpy as np
from environments.rewards.function import RewardFunction


class MarketOrderRF(RewardFunction):
    def __init__(
            self,
            timeframe_size: int,
            target_horizon_len: int,
            highs: np.ndarray,
            lows: np.ndarray,
            closes: np.ndarray,
            fees_percentage: float
    ):
        super().__init__(
            timeframe_size=timeframe_size,
            target_horizon_len=target_horizon_len,
            highs=highs,
            lows=lows,
            closes=closes,
            fees_percentage=fees_percentage
        )

    def _build_reward_fn(
            self,
            timeframe_size: int,
            target_horizon_len: int,
            highs: np.ndarray,
            lows: np.ndarray,
            closes: np.ndarray
    ) -> np.ndarray:
        return np.float32([[
            np.log(closes[i: i + target_horizon_len].max()/closes[i - 1]),
            np.log(closes[i - 1]/closes[i: i + target_horizon_len].min())
        ] for i in range(timeframe_size, closes.shape[0] - target_horizon_len + 1)])
