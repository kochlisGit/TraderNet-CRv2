import pandas as pd
from ta.momentum import StochasticOscillator
from analysis.technical.indicators.indicator import TechnicalIndicator


class STOCH(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='stoch')

    def _compute_indicator_values(
            self,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            window: int = 14
    ) -> pd.Series:
        return StochasticOscillator(high=highs, low=lows, close=closes, window=window).stoch()
