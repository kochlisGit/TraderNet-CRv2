from metrics.metric import Metric


class InvestmentRisk(Metric):
    def __init__(self):
        super().__init__(name='Investment Risk')
        self._sum_good_transactions = 0
        self._sum_bad_transactions = 0

    def reset(self):
        self._sum_good_transactions = 0
        self._sum_bad_transactions = 0

    def update(self, log_pnl: float):
        if log_pnl > 0:
            self._sum_good_transactions += 1
        elif log_pnl < 0:
            self._sum_bad_transactions += 1
        else:
            return

    def result(self) -> float:
        total_investments = (self._sum_bad_transactions + self._sum_good_transactions)
        return 0 if total_investments == 0 else self._sum_bad_transactions/total_investments
