from metrics.metric import Metric


class MaximumDrawdown(Metric):
    def __init__(self):
        super().__init__(name='Maximum Drawdown')
        self._log_pnl_sum = 0
        self._log_pnl_sum_peak = 0
        self._hourly_mdds = []

    def reset(self):
        self._log_pnl_sum = 0
        self._log_pnl_sum_peak = 0
        self._hourly_mdds = []

    def update(self, log_pnl: float):
        self._log_pnl_sum += log_pnl

        if self._log_pnl_sum_peak < self._log_pnl_sum:
            self._log_pnl_sum_peak = self._log_pnl_sum

        self._hourly_mdds.append(1 if self._log_pnl_sum_peak == 0 else self._log_pnl_sum/self._log_pnl_sum_peak)

    def result(self) -> float:
        log_mdd = min(self._hourly_mdds)
        return 1 - log_mdd
