import pandas as pd
from ta.trend import EMAIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class EMA(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='ema')

    def _compute_indicator_values(self, closes: pd.Series, window: int = 14) -> pd.Series:
        return EMAIndicator(close=closes, window=window).ema_indicator()
