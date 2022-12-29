import pandas as pd
from ta.trend import ADXIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class ADX(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='adx')

    def _compute_indicator_values(
            self,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            window: int = 14
    ) -> pd.Series:
        return ADXIndicator(high=highs, low=lows, close=closes, window=window).adx()
