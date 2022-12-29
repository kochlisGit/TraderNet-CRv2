import pandas as pd
from ta.trend import CCIIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class CCI(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='cci')

    def _compute_indicator_values(
            self,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            window: int = 20
    ) -> pd.Series:
        return CCIIndicator(
            high=highs,
            low=lows,
            close=closes,
            window=window
        ).cci()
