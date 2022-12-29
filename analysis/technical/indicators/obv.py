import pandas as pd
from ta.volume import OnBalanceVolumeIndicator
from analysis.technical.indicators.indicator import TechnicalIndicator


class OBV(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='obv')

    def _compute_indicator_values(self, closes: pd.Series, volumes: pd.Series) -> pd.Series:
        return OnBalanceVolumeIndicator(
            close=closes,
            volume=volumes
        ).on_balance_volume()
