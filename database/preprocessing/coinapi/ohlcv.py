import numpy as np
import pandas as pd
from database.preprocessing.preprocessing import DatasetPreprocessing


class OHLCVPreprocessing(DatasetPreprocessing):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _preprocess_ohlcv_columns(ohlcv_df: pd.DataFrame):
        ohlcv_df.drop(columns=['time_period_start', 'time_open', 'time_close'], inplace=True)
        ohlcv_df.rename(columns={
            'time_period_end': 'date',
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume',
            'trades_count': 'trades'
        }, inplace=True)
        ohlcv_df['date'] = ohlcv_df['date'].apply(lambda date: date.split('.')[0].replace('T', ' '))
        ohlcv_df['hour'] = ohlcv_df['date'].apply(lambda date: int(date.split(' ')[1].split(':')[0]))
        return ohlcv_df

    @staticmethod
    def _append_ohlcv_log_returns_to_df(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        ohlcv_df['open_log_returns'] = np.log(ohlcv_df['open']).diff()
        ohlcv_df['high_log_returns'] = np.log(ohlcv_df['high']).diff()
        ohlcv_df['low_log_returns'] = np.log(ohlcv_df['low']).diff()
        ohlcv_df['close_log_returns'] = np.log(ohlcv_df['close']).diff()
        ohlcv_df['volume_log_returns'] = np.log(ohlcv_df['volume']).diff()
        ohlcv_df['trades_log_returns'] = np.log(ohlcv_df['trades']).diff()
        return ohlcv_df

    def preprocess(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        ohlcv_df = self._preprocess_ohlcv_columns(ohlcv_df=ohlcv_df)
        ohlcv_df = self._append_ohlcv_log_returns_to_df(ohlcv_df=ohlcv_df)
        return ohlcv_df
