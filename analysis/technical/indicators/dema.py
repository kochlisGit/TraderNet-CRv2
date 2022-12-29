import pandas as pd
from ta.trend import EMAIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class DEMA(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='dema')

    def _compute_indicator_values(self, closes: pd.Series, window: int = 15) -> pd.Series:
        ema = EMAIndicator(close=closes, window=window).ema_indicator()
        ema_of_ema = EMAIndicator(close=ema, window=window).ema_indicator()
        return 2*ema - ema_of_ema
