import pandas as pd
from ta.trend import AroonIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class AROONS(TechnicalIndicator):
    def __init__(self):
        super().__init__(name=('aroon_up', 'aroon_down'))

    def _compute_indicator_values(self, closes: pd.Series, window: int = 25) -> tuple[pd.Series, pd.Series]:
        aroon_values = AroonIndicator(close=closes, window=window)
        return aroon_values.aroon_up(), aroon_values.aroon_down()
