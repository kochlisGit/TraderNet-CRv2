import pandas as pd
from ta.momentum import RSIIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class RSI(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='rsi')

    def _compute_indicator_values(self, closes: pd.Series, window: int = 14) -> pd.Series:
        return RSIIndicator(close=closes, window=window).rsi()
