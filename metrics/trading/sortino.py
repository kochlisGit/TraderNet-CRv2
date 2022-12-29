import numpy as np
from metrics.metric import Metric


class SortinoRatio(Metric):
    def __init__(self):
        super().__init__(name='Sortino')
        self._episode_log_pnls = []

    def reset(self):
        self._episode_log_pnls = []

    def update(self, log_pnl: float):
        self._episode_log_pnls.append(log_pnl)

    def result(self) -> float:
        episode_log_returns = np.float64(self._episode_log_pnls)
        average_returns = episode_log_returns.mean()
        std_downfall_returns = episode_log_returns[episode_log_returns < 0].std()
        return np.exp(average_returns/std_downfall_returns)
