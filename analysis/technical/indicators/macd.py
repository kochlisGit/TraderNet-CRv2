import pandas as pd
from ta.trend import MACD
from analysis.technical.indicators.indicator import TechnicalIndicator


class MACDSignalDiffs(TechnicalIndicator):
    def __init__(self):
        super().__init__(name='macd_signal_diffs')

    def _compute_indicator_values(
            self, closes: pd.Series,
            short_window: int = 12,
            long_window: int = 26,
            signal_period=9
    ) -> pd.Series:
        return MACD(
            close=closes,
            window_slow=long_window,
            window_fast=short_window,
            window_sign=signal_period
        ).macd_diff()
