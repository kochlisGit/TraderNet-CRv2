import pandas as pd
from ta.volatility import BollingerBands
from analysis.technical.indicators.indicator import TechnicalIndicator


class BBANDS(TechnicalIndicator):
    def __init__(self):
        super().__init__(name=('bband_up', 'bband_down'))

    def _compute_indicator_values(
            self, closes: pd.Series,
            window: int = 20,
            window_deviation: int = 2
    ) -> tuple[pd.Series, pd.Series]:
        bbands_values = BollingerBands(close=closes, window=window, window_dev=window_deviation)
        return bbands_values.bollinger_hband(), bbands_values.bollinger_lband()
