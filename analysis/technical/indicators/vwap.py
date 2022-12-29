import pandas as pd
from ta.volume import VolumeWeightedAveragePrice
from analysis.technical.indicators.indicator import TechnicalIndicator


class VWAP(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='vwap')

    def _compute_indicator_values(
            self,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            volumes: pd.Series,
            window: int = 10
    ) -> pd.Series:
        return VolumeWeightedAveragePrice(
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
            window=window
        ).volume_weighted_average_price()
