import pandas as pd
from ta.volume import AccDistIndexIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class ADL(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='adl')

    def _compute_indicator_values(
            self, highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            volumes: pd.Series
    ) -> pd.Series:
        return AccDistIndexIndicator(
            high=highs,
            low=lows,
            close=closes,
            volume=volumes
        ).acc_dist_index()
