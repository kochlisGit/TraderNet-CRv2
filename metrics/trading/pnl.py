from metrics.metric import Metric


class CumulativeLogReturn(Metric):
    def __init__(self):
        super().__init__(name='Cumulative Log Returns')
        self._log_pnl_sum = 0

    def reset(self):
        self._log_pnl_sum = 0

    def update(self, log_pnl: float):
        self._log_pnl_sum += log_pnl

    def result(self) -> float:
        return self._log_pnl_sum
